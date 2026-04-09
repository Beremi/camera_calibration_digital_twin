"""Dropout-debug runner and summary regressions."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from calib_sim.isaac.estimation.online_filter import AnchoredOnlineFilter
from calib_sim.isaac.logging.schemas import IsaacTagDetectionPacket
from calib_sim.isaac.tag_builder import anchor_pose_identity
from calib_sim.reporting.isaac_second_pass_dropout_debug import (
    build_second_pass_dropout_debug_bundle,
    classify_second_pass_dropout_root_cause,
    generate_second_pass_dropout_debug_artifacts,
)
from tests._isaac_test_helpers import make_second_pass_dropout_debug_runs


def test_dropout_debug_runner_dry_run_exposes_fixed_run_ids_and_debug_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_isaac_second_pass_dropout_debug.py"),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    run_ids = [entry["run_id"] for entry in payload["planned_runs"]]
    assert run_ids == [
        "second_pass_dropout_debug_visual_seed_007",
        "second_pass_dropout_debug_fused_seed_007",
        "second_pass_dropout_debug_fused_no_reacq_seed_007",
        "second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007",
        "second_pass_dropout_debug_fused_covinfl_seed_007",
        "second_pass_dropout_debug_fused_clipcorr_seed_007",
    ]
    commands = {entry["run_id"]: entry["command"] for entry in payload["planned_runs"]}
    assert "--no-allow-anchor-reacquisition-after-first-lock" in commands[
        "second_pass_dropout_debug_fused_no_reacq_seed_007"
    ]
    assert "--disable-imu-prediction-while-anchor-suppressed" in commands[
        "second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007"
    ]
    assert "--dropout-post-reacquisition-covariance-scale" in commands[
        "second_pass_dropout_debug_fused_covinfl_seed_007"
    ]
    assert "--dropout-max-reacquisition-position-correction-m" in commands[
        "second_pass_dropout_debug_fused_clipcorr_seed_007"
    ]


def test_dropout_debug_runner_accepts_split_process_and_suppression_gate_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_isaac_second_pass_dropout_debug.py"),
            "--output-root",
            str(output_root),
            "--gyro-process-covariance-scale",
            "2.0",
            "--accel-process-covariance-scale",
            "8.0",
            "--suppression-imu-specific-force-gate-mps2",
            "10.0",
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    fused_commands = [
        entry["command"]
        for entry in payload["planned_runs"]
        if entry["estimator_mode"] == "fused"
    ]
    assert fused_commands
    for command in fused_commands:
        assert "--gyro-process-covariance-scale" in command
        assert "--accel-process-covariance-scale" in command
        assert "--suppression-imu-specific-force-gate-mps2" in command


def test_dropout_debug_artifacts_and_bundle_write_expected_schema(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "isaac_runs"
    docs_path = tmp_path / "docs" / "isaac_second_pass_dropout_debug.md"
    run_dirs = make_second_pass_dropout_debug_runs(output_root)

    for run_dir in run_dirs:
        payload = generate_second_pass_dropout_debug_artifacts(run_dir)
        assert Path(payload["frames_path"]).exists()
        assert Path(payload["events_path"]).exists()
        assert (run_dir / "analysis" / "dropout_debug_summary.json").exists()
        for figure_name in (
            "dropout_pose_error_timeline.png",
            "dropout_velocity_norm_timeline.png",
            "dropout_bias_norm_timeline.png",
            "dropout_covariance_trace_timeline.png",
            "dropout_covariance_eigs_timeline.png",
            "dropout_anchor_innovation_timeline.png",
            "dropout_relocalization_correction_timeline.png",
            "dropout_visibility_schedule_overlay.png",
        ):
            assert (run_dir / "analysis" / figure_name).exists()
        with (run_dir / "raw" / "dropout_debug_frames.csv").open("r", encoding="utf-8", newline="") as handle:
            frame_headers = list(csv.DictReader(handle).fieldnames or [])
        with (run_dir / "raw" / "dropout_debug_events.csv").open("r", encoding="utf-8", newline="") as handle:
            event_headers = list(csv.DictReader(handle).fieldnames or [])
        assert "position_error_norm_m" in frame_headers
        assert "covariance_trace" in frame_headers
        assert "reason" in event_headers
        assert "relocalization_correction_norm_m" in event_headers

    bundle = build_second_pass_dropout_debug_bundle(output_root, docs_path=docs_path)
    assert Path(bundle["summary_csv"]).exists()
    assert Path(bundle["summary_json"]).exists()
    assert Path(bundle["docs_path"]).exists()

    summary = json.loads(Path(bundle["summary_json"]).read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 6
    hints = {row["run_id"]: row["root_cause_hint"] for row in summary["rows"]}
    assert hints["second_pass_dropout_debug_fused_seed_007"] == "reacquisition_update_problem"
    assert "second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007" in summary["blocker_clear_run_ids"]


def test_anchor_reacquisition_clipping_and_covariance_inflation_are_applied() -> None:
    filt = AnchoredOnlineFilter.identity_initialized(estimator_mode="fused")
    filt.dropout_post_reacquisition_covariance_scale = 8.0
    filt.dropout_max_reacquisition_position_correction_m = 0.05
    filt.dropout_max_reacquisition_rotation_correction_deg = 5.0
    filt.dropout_max_reacquisition_velocity_correction_mps = 0.20
    filt.last_anchor_position_world_m = filt.state.position_world_m.copy()
    filt.last_anchor_timestamp_s = 0.0
    pre_trace = float(filt.covariance.trace())

    result = filt.update_anchor(
        tag_detection=IsaacTagDetectionPacket(
            timestamp_s=1.0,
            sim_time_s=1.0,
            frame_index=0,
            tag_id=0,
            corners_xy=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            corner_order="tl,tr,br,bl",
            score=1.0,
            is_anchor=True,
            tag_size_m=0.2,
            pnp_tag_size_m=0.2,
            local_tag_points_m=((-0.1, -0.1, 0.0), (0.1, -0.1, 0.0), (0.1, 0.1, 0.0), (-0.1, 0.1, 0.0)),
            pose_camera_rvec=(0.0, 0.0, 0.0),
            pose_camera_tvec_m=(1.0, 0.0, 0.0),
            detector_backend="native_apriltag",
            visibility_flags={"native_backend": True, "pose_ready": True},
        ),
        tag_pose=anchor_pose_identity(0, 0.2),
        is_reacquisition=True,
    )

    assert result.accepted is True
    assert result.reason == "correction_clipped"
    assert abs(float(filt.state.position_world_m[0])) <= 0.05 + 1e-9
    assert float(result.post_update_covariance_trace) > pre_trace


def test_dropout_root_cause_classifier_handles_reacq_propagation_and_bias_cases() -> None:
    reacq_hint = classify_second_pass_dropout_root_cause(
        {
            "mean_position_error_m": 0.22,
            "empirical_95_coverage_percent": 15.0,
            "pose_nees": 1000.0,
            "max_position_error_during_suppression_m": 0.04,
            "before_reacquisition_position_error_m": 0.04,
            "after_reacquisition_position_error_m": 0.35,
            "jump_after_reacquisition_m": 0.31,
            "monotonic_during_suppression": False,
        }
    )
    propagation_hint = classify_second_pass_dropout_root_cause(
        {
            "mean_position_error_m": 0.24,
            "empirical_95_coverage_percent": 30.0,
            "pose_nees": 80.0,
            "max_position_error_during_suppression_m": 0.30,
            "jump_after_reacquisition_m": 0.02,
            "monotonic_during_suppression": True,
        }
    )
    bias_hint = classify_second_pass_dropout_root_cause(
        {
            "mean_position_error_m": 0.24,
            "empirical_95_coverage_percent": 20.0,
            "pose_nees": 200.0,
            "max_position_error_during_suppression_m": 0.30,
            "max_bias_norm": 0.5,
            "bias_growth_ratio": 25.0,
            "jump_after_reacquisition_m": 0.01,
            "monotonic_during_suppression": True,
        }
    )

    assert reacq_hint == "reacquisition_update_problem"
    assert propagation_hint == "propagation_process_problem"
    assert bias_hint == "bias_handling_problem"
