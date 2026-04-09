"""Second-pass runtime/config switch regressions."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_script_dry_run_exposes_auxiliary_isolation_switches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_root = tmp_path / "output" / "isaac_runs"
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_isaac_anchor_vio.py"),
            "--validate-config-only",
            "--output-root",
            str(output_root),
            "--run-id",
            "anchor_only_probe",
            "--estimator-mode",
            "visual",
            "--controller-mode",
            "closed-loop",
            "--no-use-aux-tags-in-filter",
            "--no-use-aux-tags-in-smoother",
            "--no-use-aux-map-for-control",
            "--smoother-backend",
            "windowed_ba",
            "--vision-covariance-scale",
            "2.0",
            "--anchor-vision-covariance-scale",
            "5.0",
            "--aux-vision-covariance-scale",
            "6.0",
            "--imu-process-covariance-scale",
            "3.0",
            "--gyro-process-covariance-scale",
            "7.0",
            "--accel-process-covariance-scale",
            "8.0",
            "--post-relocalization-covariance-scale",
            "4.0",
            "--visibility-config",
            "config/isaac/visibility/anchor_dropout_nominal.yaml",
            "--no-promote-global-latest",
        ],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    estimation_filter = payload["config_snapshots"]["estimation"]["filter"]
    estimation_smoother = payload["config_snapshots"]["estimation"]["smoother"]
    control_config = payload["config_snapshots"]["control"]

    assert estimation_filter["use_aux_tags_in_filter"] is False
    assert estimation_smoother["use_aux_tags_in_smoother"] is False
    assert estimation_smoother["backend"] == "windowed_ba"
    assert control_config["use_aux_map_for_control"] is False
    assert estimation_filter["vision_covariance_scale"] == 2.0
    assert estimation_filter["anchor_vision_covariance_scale"] == 5.0
    assert estimation_filter["aux_vision_covariance_scale"] == 6.0
    assert estimation_filter["imu_process_covariance_scale"] == 3.0
    assert estimation_filter["gyro_process_covariance_scale"] == 7.0
    assert estimation_filter["accel_process_covariance_scale"] == 8.0
    assert estimation_filter["post_relocalization_covariance_scale"] == 4.0
    assert payload["config_snapshots"]["visibility"]["name"] == "anchor_dropout_nominal"
