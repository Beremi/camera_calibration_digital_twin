"""Second-pass tuning summary and lock update regression."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests._isaac_test_helpers import make_second_pass_draft_lock


def _write_fake_tuning_run(run_dir: Path, *, position_error: float, waypoint_error: float, coverage: float, nees: float) -> None:
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    metrics = {
        "trajectory": {"mean_position_error_m": position_error},
        "control": {
            "mean_waypoint_error_m": waypoint_error,
            "completion_fraction": 1.0,
            "mean_actuator_tracking_error": 0.004,
        },
        "uncertainty_calibration": {
            "mean_position_radius_95_m": 0.028,
            "empirical_95_coverage_percent": coverage,
            "pose_nees": nees,
        },
    }
    quality = {
        "summary": {
            "anchor_mean_reprojection_rmse_px": 0.15,
            "auxiliary_mean_reprojection_rmse_px": 0.45,
            "mean_smoother_correction_norm_m": 0.02,
        }
    }
    (run_dir / "analysis" / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "analysis" / "estimator_quality.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_second_pass_tuning_runner_updates_lock_from_completed_candidates(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    docs_dir = tmp_path / "docs"
    lock_path = make_second_pass_draft_lock(docs_dir)

    run_id_a = "second_pass_tuning_anchor_only_lightweight_seed_007_imu_2p0_vision_1p0_post_1p0_gyro_default_accel_default"
    run_id_b = "second_pass_tuning_anchor_only_lightweight_seed_007_imu_4p0_vision_1p0_post_1p0_gyro_default_accel_default"
    _write_fake_tuning_run(output_root / run_id_a, position_error=0.018, waypoint_error=0.023, coverage=94.0, nees=7.0)
    _write_fake_tuning_run(output_root / run_id_b, position_error=0.016, waypoint_error=0.021, coverage=96.0, nees=5.5)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_isaac_second_pass_tuning.py"),
            "--output-root",
            str(output_root),
            "--lock-path",
            str(lock_path),
            "--imu-process-covariance-scales",
            "2",
            "4",
            "--vision-covariance-scales",
            "1",
            "--post-relocalization-covariance-scales",
            "1",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    updated_lock = json.loads(lock_path.read_text(encoding="utf-8"))

    assert payload["selected_run_id"] == run_id_b
    assert updated_lock["draft_selection"]["fused_nominal_reference_run_id"] == run_id_b
    assert updated_lock["draft_selection"]["fused_nominal_covariance_scales"]["imu_process_covariance_scale"] == 4.0
