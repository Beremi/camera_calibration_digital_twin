"""Regression coverage for the expanded first-pass suite table bundle."""

from __future__ import annotations

import json
from pathlib import Path

from calib_sim.reporting.isaac_first_pass_suite import generate_first_pass_suite_artifacts

from tests._isaac_test_helpers import make_fake_first_pass_suite


def test_first_pass_suite_writes_uncertainty_residual_and_map_tables(tmp_path: Path) -> None:
    output_root = make_fake_first_pass_suite(tmp_path / "output" / "isaac_runs")
    generate_first_pass_suite_artifacts(output_root, regenerate_run_artifacts=False)

    suite_dir = output_root / "latest_first_pass_suite"
    analysis_dir = suite_dir / "analysis"
    summary = json.loads((analysis_dir / "suite_summary.json").read_text(encoding="utf-8"))

    assert summary["uncertainty"][0]["mean_position_radius_95_m"] is not None
    assert summary["residuals"][0]["mean_reprojection_rmse_px"] is not None
    assert summary["map_quality"][0]["mean_auxiliary_tag_position_error_m"] is not None

    for path in (
        analysis_dir / "first_pass_uncertainty_table.csv",
        analysis_dir / "first_pass_residual_table.csv",
        analysis_dir / "first_pass_map_quality_table.csv",
    ):
        assert path.exists()

    paper_artifacts = (analysis_dir / "report_data" / "paper_artifacts.tex").read_text(encoding="utf-8")
    assert r"\IsaacFirstPassUncertaintyRows" in paper_artifacts
    assert r"\IsaacFirstPassResidualRows" in paper_artifacts
    assert r"\IsaacFirstPassMapRows" in paper_artifacts
