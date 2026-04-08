"""Table generation for Isaac run reports."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: Any) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def _tex_value(value: Any, *, digits: int = 3, percent: bool = False, na: bool = False) -> str:
    if na:
        return "n/a"
    if value in ("", None):
        return r"\ArtifactPending{}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        formatted = f"{value:.{digits}f}"
        if percent:
            return _tex_escape(formatted)
        return formatted
    return _tex_escape(value)


def _write_tex_rows(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _macro_definition(name: str, rows: list[str]) -> str:
    body = "\n".join(rows)
    return "\n".join(
        [
            rf"\providecommand{{\{name}}}{{}}",
            rf"\renewcommand{{\{name}}}{{%",
            body,
            "}",
        ]
    )


def write_isaac_report_tables(run_dir: str | Path, metrics: dict[str, Any]) -> dict[str, str]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    report_data_dir = analysis_dir / "report_data"
    report_data_dir.mkdir(parents=True, exist_ok=True)

    run_description_rows = [
        {"quantity": "run_id", "value": metrics["run_id"]},
        {"quantity": "isaac_sim_version", "value": metrics["manifest"]["isaac_sim_version"]},
        {"quantity": "physics_rate_hz", "value": metrics["config"]["physics_rate_hz"]},
        {"quantity": "imu_rate_hz", "value": metrics["config"]["imu_rate_hz"]},
        {"quantity": "camera_rate_hz", "value": metrics["config"]["camera_rate_hz"]},
        {"quantity": "frames", "value": metrics["counts"]["camera_frames"]},
        {"quantity": "imu_packets", "value": metrics["counts"]["imu_packets"]},
        {"quantity": "observed_auxiliary_tags", "value": metrics["counts"]["unique_detected_auxiliary_tags"]},
        {"quantity": "anchor_tag_id", "value": metrics["manifest"]["anchor_tag_id"]},
        {"quantity": "noise_preset", "value": metrics["config"]["noise_preset"]},
        {"quantity": "seed", "value": metrics["manifest"]["random_seed"]},
    ]
    estimate_summary_rows = [
        {
            "mean_position_radius_95_m": metrics["estimation"]["mean_position_radius_95_m"],
            "max_position_radius_95_m": metrics["estimation"]["max_position_radius_95_m"],
            "mean_innovation_norm": metrics["estimation"]["mean_innovation_norm"],
            "saturation_fraction": metrics["control"]["saturation_fraction"],
        }
    ]
    run_description_path = analysis_dir / "run_description_table.csv"
    estimate_summary_path = analysis_dir / "estimate_summary_table.csv"
    _write_dict_rows(run_description_path, run_description_rows)
    _write_dict_rows(estimate_summary_path, estimate_summary_rows)

    run_description_tex_rows = [
        rf"Run id & {_tex_value(metrics['run_id'])} \\",
        rf"Isaac Sim version & {_tex_value(metrics['manifest']['isaac_sim_version'])} \\",
        rf"Physics rate [Hz] & {_tex_value(metrics['config']['physics_rate_hz'], digits=1)} \\",
        rf"IMU rate [Hz] & {_tex_value(metrics['config']['imu_rate_hz'], digits=1)} \\",
        rf"Camera rate [Hz] & {_tex_value(metrics['config']['camera_rate_hz'], digits=1)} \\",
        rf"Frames & {_tex_value(metrics['counts']['camera_frames'])} \\",
        rf"IMU packets & {_tex_value(metrics['counts']['imu_packets'])} \\",
        rf"Observed auxiliary tags & {_tex_value(metrics['counts']['unique_detected_auxiliary_tags'])} \\",
        rf"Anchor tag id & {_tex_value(metrics['manifest']['anchor_tag_id'])} \\",
        rf"Noise preset & {_tex_value(metrics['config']['noise_preset'])} \\",
        rf"Seed & {_tex_value(metrics['manifest']['random_seed'])} \\",
    ]
    trajectory_map_tex_rows = [
        r"Visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        rf"Fused & {_tex_value(metrics['trajectory']['mean_position_error_m'])} & {_tex_value(metrics['trajectory']['p95_position_error_m'])} & {_tex_value(metrics['trajectory']['max_position_error_m'])} & {_tex_value(metrics['trajectory']['mean_rotation_error_deg'])} & {_tex_value(metrics['trajectory']['mean_reprojection_error_px'])} & {_tex_value(metrics['trajectory']['map_error_m'])} \\",
        rf"IMU-only & {_tex_value(metrics['trajectory']['mean_position_error_m'])} & {_tex_value(metrics['trajectory']['p95_position_error_m'])} & {_tex_value(metrics['trajectory']['max_position_error_m'])} & {_tex_value(metrics['trajectory']['mean_rotation_error_deg'])} & n/a & n/a \\",
        r"Known-map diagnostic & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & n/a \\",
    ]
    path_tracking_tex_rows = [
        r"Open-loop commands & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Closed-loop visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        rf"Closed-loop fused & {_tex_value(metrics['control']['mean_waypoint_error_m'])} & {_tex_value(metrics['control']['p95_waypoint_error_m'])} & {_tex_value(None if metrics['control']['completion_fraction'] is None else metrics['control']['completion_fraction'] * 100.0, percent=True)} & {_tex_value(metrics['control']['mean_completion_time_s'])} & {_tex_value(metrics['control']['saturation_fraction'] * 100.0, percent=True)} & {_tex_value(metrics['control']['failure_rate_percent'], percent=True)} \\",
    ]
    parameter_plausibility_tex_rows = [
        rf"Fused & {_tex_value(metrics['parameters']['mean_gyro_bias_norm_rps'])} & {_tex_value(metrics['parameters']['mean_accel_bias_norm_mps2'])} & {_tex_value(metrics['residuals']['mean_visual_residual_sq'])} & {_tex_value(metrics['residuals']['mean_imu_residual_sq'])} & {_tex_value(metrics['control']['mean_actuator_tracking_error'])} & {_tex_value('single-run artifact summary')} \\",
    ]
    uncertainty_calibration_tex_rows = [
        r"Visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & n/a & \ArtifactPending{} \\",
        rf"Fused & {_tex_value(metrics['uncertainty_calibration']['mean_position_radius_95_m'])} & {_tex_value(metrics['uncertainty_calibration']['empirical_95_coverage_percent'], percent=True)} & {_tex_value(metrics['uncertainty_calibration']['pose_nees'])} & {_tex_value(metrics['uncertainty_calibration']['velocity_nees'])} & {_tex_value(metrics['uncertainty_calibration']['sigma_error_correlation'])} \\",
    ]
    noise_sensitivity_tex_rows = [
        r"Ideal & Visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Ideal & Fused & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Nominal & Visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Nominal & Fused & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Stress & Visual-only & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
        r"Stress & Fused & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} & \ArtifactPending{} \\",
    ]

    _write_tex_rows(report_data_dir / "run_description_rows.tex", run_description_tex_rows)
    _write_tex_rows(report_data_dir / "trajectory_map_rows.tex", trajectory_map_tex_rows)
    _write_tex_rows(report_data_dir / "path_tracking_rows.tex", path_tracking_tex_rows)
    _write_tex_rows(report_data_dir / "parameter_plausibility_rows.tex", parameter_plausibility_tex_rows)
    _write_tex_rows(report_data_dir / "uncertainty_calibration_rows.tex", uncertainty_calibration_tex_rows)
    _write_tex_rows(report_data_dir / "noise_sensitivity_rows.tex", noise_sensitivity_tex_rows)
    (report_data_dir / "estimate_summary.tex").write_text(
        "\n".join(
            [
                rf"\newcommand{{\IsaacMeanPositionRadius}}{{{metrics['estimation']['mean_position_radius_95_m']}}}",
                rf"\newcommand{{\IsaacMaxPositionRadius}}{{{metrics['estimation']['max_position_radius_95_m']}}}",
                rf"\newcommand{{\IsaacMeanInnovationNorm}}{{{metrics['estimation']['mean_innovation_norm']}}}",
                rf"\newcommand{{\IsaacSaturationFraction}}{{{metrics['control']['saturation_fraction']}}}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_data_dir / "paper_artifacts.tex").write_text(
        "\n\n".join(
            [
                _macro_definition("IsaacRunDescriptionRows", run_description_tex_rows),
                _macro_definition("IsaacTrajectoryMapRows", trajectory_map_tex_rows),
                _macro_definition("IsaacPathTrackingRows", path_tracking_tex_rows),
                _macro_definition("IsaacParameterPlausibilityRows", parameter_plausibility_tex_rows),
                _macro_definition("IsaacUncertaintyCalibrationRows", uncertainty_calibration_tex_rows),
                _macro_definition("IsaacNoiseSensitivityRows", noise_sensitivity_tex_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "run_description_table": str(run_description_path),
        "estimate_summary_table": str(estimate_summary_path),
        "run_description_rows_tex": str(report_data_dir / "run_description_rows.tex"),
        "trajectory_map_rows_tex": str(report_data_dir / "trajectory_map_rows.tex"),
        "path_tracking_rows_tex": str(report_data_dir / "path_tracking_rows.tex"),
        "parameter_plausibility_rows_tex": str(report_data_dir / "parameter_plausibility_rows.tex"),
        "uncertainty_calibration_rows_tex": str(report_data_dir / "uncertainty_calibration_rows.tex"),
        "noise_sensitivity_rows_tex": str(report_data_dir / "noise_sensitivity_rows.tex"),
        "estimate_summary_tex": str(report_data_dir / "estimate_summary.tex"),
        "paper_artifacts_tex": str(report_data_dir / "paper_artifacts.tex"),
    }
