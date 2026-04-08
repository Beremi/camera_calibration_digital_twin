"""Suite-level aggregation checks for the first-pass Isaac benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from calib_sim.reporting.isaac_first_pass_suite import generate_first_pass_suite_artifacts

from tests._isaac_test_helpers import make_fake_first_pass_suite


def test_first_pass_suite_generation_writes_tables_macros_and_summary_sections(tmp_path: Path) -> None:
    output_root = make_fake_first_pass_suite(tmp_path / "output" / "isaac_runs")

    payload = generate_first_pass_suite_artifacts(output_root, regenerate_run_artifacts=False)

    suite_dir = output_root / "latest_first_pass_suite"
    assert payload["canonical_run_id"] == "first_pass_fused_closed-loop_servo_nominal_seed_007"
    summary = json.loads((suite_dir / "analysis" / "suite_summary.json").read_text(encoding="utf-8"))
    assert "uncertainty" in summary
    assert "residuals" in summary
    assert "map_quality" in summary
    assert summary["publication"]["artifact_source"] == "latest_first_pass_suite"

    tex = (suite_dir / "analysis" / "report_data" / "paper_artifacts.tex").read_text(encoding="utf-8")
    for macro in (
        "IsaacFirstPassMatrixRows",
        "IsaacFirstPassReproducibilityRows",
        "IsaacFirstPassUncertaintyRows",
        "IsaacFirstPassResidualRows",
        "IsaacFirstPassMapRows",
    ):
        assert macro in tex
    assert r"servo\_nominal" in tex
    assert r"\ArtifactPending{}" not in tex

    for csv_name in (
        "first_pass_matrix_table.csv",
        "first_pass_uncertainty_table.csv",
        "first_pass_residual_table.csv",
        "first_pass_map_quality_table.csv",
    ):
        assert (suite_dir / "analysis" / csv_name).exists()
    assert (suite_dir / "analysis" / "report_data" / "ik_failure_timeline.png").exists()
