"""Regression checks for generated Checkpoint 03 headline artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUN_DIR = Path("output/interactive_runs/run_20260407_143339")
ANALYSIS_DIR = RUN_DIR / "analysis"


def test_checkpoint03_summary_captures_headline_regime() -> None:
    summary = json.loads((ANALYSIS_DIR / "summary.json").read_text(encoding="utf-8"))

    assert summary["schema_version"] == 4
    assert summary["batch_v2"]["plausibility"]["accel_bias_norm_mean_pass"] is True
    assert summary["batch_v2"]["mean_position_error_m"] <= summary["batch_v1"]["mean_position_error_m"]
    assert summary["batch_v2"]["mean_rotation_error_deg"] <= summary["batch_v1"]["mean_rotation_error_deg"]
    assert summary["factor_breakdown"]["fused"]["mean_whitened_imu_sq_residual_per_factor"] < 12.0
    assert summary["ablations"]["sweep_seeds"] == [11, 17, 23, 31, 47]
    assert summary["ablations"]["nominal"]["fused"]["mean_position_error_m"] < summary["ablations"]["nominal"]["visual_only"]["mean_position_error_m"]
    assert summary["ablations"]["stress"]["fused"]["mean_position_error_m"] < summary["ablations"]["stress"]["visual_only"]["mean_position_error_m"]


def test_checkpoint03_scientific_report_contains_canonical_sections() -> None:
    report = (ANALYSIS_DIR / "checkpoint_03_scientific_report.md").read_text(encoding="utf-8")

    expected_sections = [
        "## 1. Problem Statement",
        "## 2. Benchmark Context",
        "## 3. Canonical Estimation Problem",
        "### 3.1 Unknown State",
        "### 3.2 Observations",
        "### 3.3 Forward Models",
        "### 3.4 Synthetic Data-Generation Noise",
        "### 3.5 Estimator Likelihood Noise",
        "### 3.6 Priors / Process Model",
        "### 3.7 MAP Objective",
        "### 3.8 Posterior Uncertainty Extraction",
        "### 3.9 Uncertainty Evaluation",
        "### Block D: Noise Sensitivity",
    ]
    positions = [report.index(section) for section in expected_sections]
    assert positions == sorted(positions)
    assert "Initialization seeds LM but is not itself a prior term." in report
    assert "Huber is robustification applied after visual whitening; it is not the Gaussian likelihood." in report
    assert "A scalar 95% position radius is only a coarse summary." in report


def test_checkpoint03_noise_sensitivity_table_reports_five_seed_sweep() -> None:
    with (ANALYSIS_DIR / "noise_sensitivity_table.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    nominal_fused = next(row for row in rows if row["noise_preset"] == "nominal" and row["method"] == "fused")
    stress_fused = next(row for row in rows if row["noise_preset"] == "stress" and row["method"] == "fused")

    assert nominal_fused["sample_count"] == "5"
    assert stress_fused["sample_count"] == "5"
    assert float(nominal_fused["mean_position_error_m"]) < float(
        next(row for row in rows if row["noise_preset"] == "nominal" and row["method"] == "visual_only")["mean_position_error_m"]
    )
    assert float(stress_fused["mean_position_error_m"]) < float(
        next(row for row in rows if row["noise_preset"] == "stress" and row["method"] == "visual_only")["mean_position_error_m"]
    )


def test_checkpoint03_likelihood_sweep_and_uncertainty_exports_have_expected_schema() -> None:
    with (ANALYSIS_DIR / "likelihood_sweep_table.csv").open("r", encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    families = sorted({row["sweep_family"] for row in sweep_rows})
    scales_by_family = {
        family: sorted(float(row["scale"]) for row in sweep_rows if row["sweep_family"] == family)
        for family in families
    }
    assert families == ["imu_covariance_scale", "visual_weight_scale"]
    assert scales_by_family["imu_covariance_scale"] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    assert scales_by_family["visual_weight_scale"] == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

    with (ANALYSIS_DIR / "uncertainty_calibration_table.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    assert "velocity_nees_mean" in headers
    assert "gyro_bias_nees" in headers
    assert "accel_bias_nees" in headers
    fused_row = next(row for row in rows if row["method"] == "fused")
    assert fused_row["velocity_nees_mean"] != ""
    assert fused_row["gyro_bias_nees"] != ""
    assert fused_row["accel_bias_nees"] != ""

    with (ANALYSIS_DIR / "uncertainty_stratified_table.csv").open("r", encoding="utf-8", newline="") as handle:
        stratified_rows = list(csv.DictReader(handle))
    assert any(row["method"] == "fused" and row["stratum_type"] == "motion_regime" for row in stratified_rows)
    assert any(row["method"] == "visual_only" and row["stratum_type"] == "visible_tags" for row in stratified_rows)

    factor_breakdown = json.loads((ANALYSIS_DIR / "factor_breakdown.json").read_text(encoding="utf-8"))
    assert "mean_whitened_imu_sq_residual_position_per_factor" in factor_breakdown["fused"]
    assert "mean_whitened_imu_sq_residual_rotation_per_factor" in factor_breakdown["fused"]
    assert "mean_whitened_imu_sq_residual_velocity_per_factor" in factor_breakdown["fused"]
    assert len(factor_breakdown["likelihood_sweeps"]["imu_covariance_scale"]) == 6
    assert len(factor_breakdown["likelihood_sweeps"]["visual_weight_scale"]) == 6

    with (ANALYSIS_DIR / "parameter_plausibility_table.csv").open("r", encoding="utf-8", newline="") as handle:
        plausibility_reader = csv.DictReader(handle)
        plausibility_headers = plausibility_reader.fieldnames or []
        plausibility_rows = list(plausibility_reader)
    assert "mean_whitened_imu_sq_residual_position_per_factor" in plausibility_headers
    assert "mean_whitened_imu_sq_residual_rotation_per_factor" in plausibility_headers
    assert "mean_whitened_imu_sq_residual_velocity_per_factor" in plausibility_headers
    assert next(row for row in plausibility_rows if row["method"] == "fused")["comment"] == "pass"

    assert (ANALYSIS_DIR / "batch_v2_uncertainty_vs_error.png").exists()
    assert (ANALYSIS_DIR / "batch_v2_imu_residual_components.png").exists()
