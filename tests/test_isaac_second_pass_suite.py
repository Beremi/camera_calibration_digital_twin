"""Second-pass draft suite aggregation regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from calib_sim.reporting.isaac_second_pass_suite import generate_second_pass_suite_artifacts

from tests._isaac_test_helpers import make_second_pass_draft_lock, make_second_pass_draft_suite_runs


def test_second_pass_suite_artifacts_write_tables_macros_and_figures(tmp_path: Path) -> None:
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    make_second_pass_draft_suite_runs(output_root)
    lock_path = make_second_pass_draft_lock(docs_dir)

    payload = generate_second_pass_suite_artifacts(output_root, lock_path=lock_path, regenerate_run_artifacts=False)
    suite_dir = Path(payload["suite_dir"])
    analysis_dir = suite_dir / "analysis"
    report_data_dir = analysis_dir / "report_data"

    assert (analysis_dir / "draft_runtime_table.csv").exists()
    assert (analysis_dir / "draft_nominal_table.csv").exists()
    assert (analysis_dir / "draft_dropout_table.csv").exists()
    assert (analysis_dir / "draft_actuation_stress_table.csv").exists()
    assert (analysis_dir / "draft_uncertainty_table.csv").exists()
    assert (analysis_dir / "draft_map_quality_table.csv").exists()

    paper_artifacts = (report_data_dir / "second_pass_paper_artifacts.tex").read_text(encoding="utf-8")
    assert r"\IsaacSecondPassRuntimeRows" in paper_artifacts
    assert r"\IsaacSecondPassNominalRows" in paper_artifacts
    assert r"\IsaacSecondPassDropoutRows" in paper_artifacts
    assert r"\IsaacSecondPassActuationStressRows" in paper_artifacts
    assert r"\IsaacSecondPassUncertaintyRows" in paper_artifacts
    assert r"\IsaacSecondPassMapQualityRows" in paper_artifacts
    assert r"\ArtifactPending{}" not in paper_artifacts
    assert "8881.9" not in paper_artifacts

    for figure_name in (
        "nominal_trajectory_compare.png",
        "dropout_trajectory_compare.png",
        "stress_trajectory_compare.png",
        "coverage_nees_compare.png",
        "anchor_vs_aux_residuals_nominal.png",
        "anchor_vs_aux_residuals_dropout.png",
        "smoother_feedback_compare.png",
        "tuning_heatmap_fused.png",
        "representative_nominal_first_frame.png",
        "representative_nominal_sim_waypoints.png",
        "representative_nominal_position_estimation.png",
        "representative_nominal_pattern_world_positions.png",
        "representative_nominal_imu_measurements.png",
        "representative_nominal_pattern_relative_camera_positions.png",
    ):
        assert (analysis_dir / figure_name).exists()

    summary = json.loads((analysis_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert len(summary["run_ids"]) == 18
    assert summary["artifact_source"] == "latest_second_pass_suite"
    assert summary["fused_filter_overrides"]["suppression_propagation_mode"] == "gyro_only"
    assert summary["fused_filter_overrides"]["suppression_imu_specific_force_gate_mps2"] == 10.0
    assert summary["fused_filter_overrides"]["dropout_post_reacquisition_covariance_scale"] == 8.0


def test_second_pass_suite_dry_run_shows_fused_filter_overrides(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    lock_path = make_second_pass_draft_lock(docs_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_isaac_second_pass_draft_suite.py"),
            "--output-root",
            str(output_root),
            "--lock-path",
            str(lock_path),
            "--dry-run",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    fused_row = next(row for row in payload["planned_runs"] if row["estimator_mode"] == "fused")

    assert fused_row["filter_overrides"]["suppression_propagation_mode"] == "gyro_only"
    assert fused_row["filter_overrides"]["suppression_imu_specific_force_gate_mps2"] == 10.0
    assert fused_row["filter_overrides"]["dropout_post_reacquisition_covariance_scale"] == 8.0
