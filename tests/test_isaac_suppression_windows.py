"""Suppression-window analysis helper regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from calib_sim.reporting.isaac_suppression_windows import (
    classify_suppression_window,
    generate_suppression_window_artifacts,
)
from tests._isaac_test_helpers import make_second_pass_dropout_debug_runs


def test_generate_suppression_window_artifacts_writes_expected_summary(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "isaac_runs"
    run_dirs = make_second_pass_dropout_debug_runs(output_root)
    fused_run_dir = next(run_dir for run_dir in run_dirs if run_dir.name == "second_pass_dropout_debug_fused_seed_007")

    payload = generate_suppression_window_artifacts(fused_run_dir)

    assert Path(payload["summary_csv"]).exists()
    assert Path(payload["summary_json"]).exists()
    assert Path(payload["note_path"]).exists()
    summary = json.loads(Path(payload["summary_json"]).read_text(encoding="utf-8"))
    assert summary["window_count"] == 1
    assert summary["dominant_classifier"] == "mean_state_drift_dominant"
    window = summary["windows"][0]
    assert window["position_error_growth_m"] > 0.0
    assert window["imu_packets_used_for_prediction"] == 18
    assert window["imu_packets_rejected_for_prediction"] == 0
    assert window["first_reacquisition_innovation_norm_m"] == 0.32
    assert window["first_reacquisition_nis"] == 12.0
    assert window["window_classifier"] == "mean_state_drift_dominant"


def test_classify_suppression_window_distinguishes_bias_reacq_and_covariance_cases() -> None:
    assert (
        classify_suppression_window(
            {
                "position_error_growth_m": 0.02,
                "post_reacquisition_position_error_jump_m": 0.12,
                "gyro_bias_growth_ratio": 1.0,
                "accel_bias_growth_ratio": 1.0,
                "covariance_trace_growth": 0.001,
            }
        )
        == "reacquisition_jump_dominant"
    )
    assert (
        classify_suppression_window(
            {
                "position_error_growth_m": 0.09,
                "post_reacquisition_position_error_jump_m": 0.0,
                "gyro_bias_growth_ratio": 7.0,
                "accel_bias_growth_ratio": 2.0,
                "covariance_trace_growth": 0.003,
            }
        )
        == "bias_growth_dominant"
    )
    assert (
        classify_suppression_window(
            {
                "position_error_growth_m": 0.0,
                "post_reacquisition_position_error_jump_m": 0.0,
                "gyro_bias_growth_ratio": 1.0,
                "accel_bias_growth_ratio": 1.0,
                "covariance_trace_growth": 0.01,
            }
        )
        == "covariance_only"
    )


def test_suppression_window_script_invokes_helper(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    run_dirs = make_second_pass_dropout_debug_runs(output_root)
    fused_run_dir = next(run_dir for run_dir in run_dirs if run_dir.name == "second_pass_dropout_debug_fused_seed_007")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_isaac_suppression_windows.py"),
            str(fused_run_dir),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["window_count"] == 1
    assert Path(payload["summary_json"]).exists()
