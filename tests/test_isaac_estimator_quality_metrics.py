"""Estimator-quality diagnostic bundle regressions."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from calib_sim.reporting import compute_isaac_estimator_quality

from tests._isaac_test_helpers import make_estimator_quality_isaac_run


def test_estimator_quality_metrics_split_anchor_aux_and_emit_bundle(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = make_estimator_quality_isaac_run(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "analyze_isaac_estimator_quality.py"),
            str(run_dir),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    summary = payload["summary"]

    assert summary["anchor_mean_reprojection_rmse_px"] is not None
    assert summary["auxiliary_mean_reprojection_rmse_px"] is not None
    assert summary["native_mean_reprojection_rmse_px"] is not None
    assert summary["fallback_mean_reprojection_rmse_px"] is not None
    assert summary["accepted_auxiliary_updates"] == 2
    assert summary["rejected_auxiliary_updates"] == 1
    assert summary["mean_smoother_correction_norm_m"] is not None

    analysis_dir = run_dir / "analysis"
    assert (analysis_dir / "estimator_quality.json").exists()
    assert (analysis_dir / "estimator_quality.csv").exists()
    assert (analysis_dir / "tag_update_breakdown.csv").exists()
    assert (analysis_dir / "anchor_vs_aux_residuals.png").exists()
    assert (analysis_dir / "smoother_correction_timeline.png").exists()

    csv_rows = list(csv.DictReader((analysis_dir / "tag_update_breakdown.csv").open("r", encoding="utf-8", newline="")))
    assert any(int(row["tag_id"]) == 43 and int(row["fallback_only"]) == 1 for row in csv_rows)


def test_estimator_quality_metrics_function_reports_per_tag_breakdown(tmp_path: Path) -> None:
    run_dir = make_estimator_quality_isaac_run(tmp_path)
    payload = compute_isaac_estimator_quality(run_dir)

    assert len(payload["tag_residuals"]) >= 2
    assert any(int(row["tag_id"]) == 42 for row in payload["tag_residuals"])
    assert any(int(row["tag_id"]) == 43 for row in payload["tag_update_breakdown"])
    assert payload["summary"]["anchor_visible_fraction"] == 1.0
