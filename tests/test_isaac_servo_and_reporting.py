"""Servo corruption and report-generation checks."""

from __future__ import annotations

import json
import pytest

from calib_sim.isaac.actuation.servo_model import ServoCorruptionConfig, UncertainServoModel
from calib_sim.reporting import generate_isaac_report_artifacts

from tests._isaac_test_helpers import make_minimal_isaac_run


def test_servo_corruption_measurably_changes_realized_motion() -> None:
    nominal = UncertainServoModel(ServoCorruptionConfig(lag_time_constant_s=0.01, rate_limit_per_s=100.0))
    corrupted = UncertainServoModel(
        ServoCorruptionConfig(
            delay_s=0.05,
            lag_time_constant_s=0.20,
            deadband=0.10,
            backlash=0.05,
            rate_limit_per_s=1.0,
        )
    )
    nominal_result = nominal.step(command=1.0, dt_s=0.1)
    corrupted_result = corrupted.step(command=1.0, dt_s=0.1)

    assert nominal_result.realized_position != corrupted_result.realized_position
    assert corrupted_result.realized_position < nominal_result.realized_position


def test_report_generation_writes_metrics_tables_and_figures(tmp_path) -> None:
    run_dir = make_minimal_isaac_run(tmp_path)
    payload = generate_isaac_report_artifacts(run_dir, allow_incomplete=True)

    metrics_path = run_dir / "analysis" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["counts"]["camera_frames"] == 1
    assert metrics["counts"]["imu_packets"] == 2
    assert metrics["counts"]["controller_diagnostics"] == 1
    assert payload["complete"] is False
    assert payload["latest_any_link"] is not None
    assert payload["latest_complete_link"] is None
    assert (run_dir / "raw" / "controller_diagnostics.csv").exists()
    assert (run_dir / "analysis" / "run_description_table.csv").exists()
    assert (run_dir / "analysis" / "estimate_summary_table.csv").exists()
    assert (run_dir / "analysis" / "report_data" / "run_description_rows.tex").exists()
    assert (run_dir / "analysis" / "report_data" / "trajectory_map_rows.tex").exists()
    assert (run_dir / "analysis" / "report_data" / "path_tracking_rows.tex").exists()
    assert (run_dir / "analysis" / "report_data" / "system_architecture.png").exists()
    assert (run_dir / "analysis" / "report_data" / "actuator_command_vs_realized.png").exists()
    assert (run_dir / "analysis" / "isaac_metrics_summary.png").exists()
    assert (run_dir / "analysis" / "isaac_uncertainty_timeline.png").exists()
    assert payload["metrics"]["run_id"] == "test_run"


def test_report_generation_requires_allow_incomplete_for_partial_runs(tmp_path) -> None:
    run_dir = make_minimal_isaac_run(tmp_path)

    with pytest.raises(ValueError, match="incomplete"):
        generate_isaac_report_artifacts(run_dir)
