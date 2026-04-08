"""Second-pass isolation bundle aggregation regressions."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from tests._isaac_test_helpers import make_second_pass_isolation_runs


def test_second_pass_isolation_bundle_writes_fixed_columns_and_summary(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    make_second_pass_isolation_runs(output_root)
    docs_path = tmp_path / "docs" / "isaac_second_pass_isolation_summary.md"

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_isaac_second_pass_isolation_bundle.py"),
            "--output-root",
            str(output_root),
            "--docs-path",
            str(docs_path),
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    bundle_dir = output_root / "latest_second_pass_isolation"
    csv_rows = list(csv.DictReader((bundle_dir / "summary.csv").open("r", encoding="utf-8", newline="")))

    assert len(csv_rows) == 6
    expected_columns = {
        "run_id",
        "estimator_mode",
        "controller_mode",
        "use_aux_tags_in_filter",
        "use_aux_tags_in_smoother",
        "use_aux_map_for_control",
        "mean_position_error_m",
        "mean_waypoint_error_m",
        "anchor_rmse_px",
        "aux_rmse_px",
        "native_pnp_fraction",
        "fallback_only_fraction",
        "aux_update_accept_fraction",
        "mean_smoother_feedback_norm_m",
        "empirical_95_coverage_percent",
        "pose_nees",
        "mean_auxiliary_tag_position_error_m",
    }
    assert expected_columns.issubset(set(csv_rows[0].keys()))
    assert payload["classification"]["classification"] in {
        "metric_projection_bug",
        "auxiliary_tag_path_bug",
        "fused_mechanization_or_weighting_bug",
        "smoother_feedback_corruption",
        "inconclusive",
    }
    assert docs_path.exists()
    markdown = docs_path.read_text(encoding="utf-8")
    assert "second_pass_visual_closed_loop_anchor_only_seed_007" in markdown
    assert "Root-cause classification" in markdown

