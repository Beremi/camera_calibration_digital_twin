"""Aggregate first-pass Isaac benchmark runs into suite-level report artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from calib_sim.reporting import generate_isaac_report_artifacts


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "analysis" / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return generate_isaac_report_artifacts(run_dir, allow_incomplete=False)["metrics"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tex_value(value: Any, *, digits: int = 3, percent: bool = False) -> str:
    if value in ("", None):
        return r"\ArtifactPending{}"
    if isinstance(value, str):
        return value.replace("_", r"\_")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if percent:
            return f"{value * 100.0:.1f}"
        return f"{value:.{digits}f}"
    return str(value)


def _macro_definition(name: str, rows: list[str]) -> str:
    return "\n".join(
        [
            rf"\providecommand{{\{name}}}{{}}",
            rf"\renewcommand{{\{name}}}{{%",
            *rows,
            "}",
        ]
    )


def _save_text_figure(path: Path, *, title: str, lines: list[str]) -> None:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 26, 32), 2, cv2.LINE_AA)
    y = 90
    for line in lines:
        cv2.putText(canvas, line[:110], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (62, 66, 72), 1, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(path), canvas)


def _save_line_plot(path: Path, *, title: str, x_values: list[float], y_values: list[float], footer: str) -> None:
    if not x_values or not y_values or len(x_values) != len(y_values):
        _save_text_figure(path, title=title, lines=["No controller-diagnostic samples available."])
        return
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    left, right, top, bottom = 90, 40, 60, 70
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, footer, (24, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (1080 - right, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0
    points = []
    for x_value, y_value in zip(xs, ys):
        x_px = origin_x + int(round((x_value - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((y_value - min_y) / (max_y - min_y) * plot_height))
        points.append((x_px, y_px))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (188, 92, 60), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None, None
    array = np.asarray(usable, dtype=np.float64)
    return float(np.mean(array)), float(np.std(array))


def _format_mean_std(mean: float | None, std: float | None, *, percent: bool = False) -> str:
    if mean is None or std is None:
        return r"\ArtifactPending{}"
    if percent:
        return rf"{mean * 100.0:.1f} $\pm$ {std * 100.0:.1f}"
    return rf"{mean:.3f} $\pm$ {std:.3f}"


def _copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _controller_diagnostic_figure(canonical_run_dir: Path, output_path: Path, controller_metrics: dict[str, Any]) -> None:
    diagnostic_path = canonical_run_dir / "raw" / "controller_diagnostics.csv"
    if not diagnostic_path.exists():
        _save_text_figure(output_path, title="IK Failure Timeline", lines=["Missing controller diagnostics log."])
        return
    with diagnostic_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    xs = [float(row["timestamp_s"]) for row in rows if row.get("timestamp_s")]
    ys = [0.0 if str(row.get("ik_success", "")).lower() in {"true", "1"} else 1.0 for row in rows]
    footer = f"dominant reason: {controller_metrics.get('dominant_safety_reason', 'n/a')}"
    _save_line_plot(output_path, title="IK Failure Timeline", x_values=xs, y_values=ys, footer=footer)


def generate_first_pass_suite_artifacts(
    output_root: str | Path,
    *,
    matrix_seed: int = 7,
    reproducibility_seeds: tuple[int, ...] = (11, 17, 23, 31, 47),
    regenerate_run_artifacts: bool = True,
    artifact_source: str = "latest_first_pass_suite",
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    suite_dir = root / "latest_first_pass_suite"
    analysis_dir = suite_dir / "analysis"
    report_data_dir = analysis_dir / "report_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    matrix_specs = [
        ("visual", "open-loop", "servo_nominal", matrix_seed),
        ("visual", "closed-loop", "servo_nominal", matrix_seed),
        ("fused", "open-loop", "servo_nominal", matrix_seed),
        ("fused", "closed-loop", "servo_nominal", matrix_seed),
    ]
    reproducibility_specs = [
        (estimator_mode, "closed-loop", "servo_nominal", seed)
        for estimator_mode in ("visual", "fused")
        for seed in reproducibility_seeds
    ]
    actuation_specs = [
        ("fused", "closed-loop", "none", matrix_seed),
        ("fused", "closed-loop", "servo_nominal", matrix_seed),
    ]
    all_specs = list(dict.fromkeys(matrix_specs + reproducibility_specs + actuation_specs))

    runs: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for estimator_mode, controller_mode, actuation_name, seed in all_specs:
        run_id = f"first_pass_{estimator_mode}_{controller_mode}_{actuation_name}_seed_{seed:03d}"
        run_dir = root / run_id
        if not run_dir.exists():
            raise ValueError(f"Required first-pass benchmark run is missing: {run_dir}")
        if regenerate_run_artifacts or not (run_dir / "analysis" / "metrics.json").exists():
            generate_isaac_report_artifacts(run_dir, allow_incomplete=False)
        runs[(estimator_mode, controller_mode, actuation_name, seed)] = _load_metrics(run_dir)

    canonical_metrics = runs[("fused", "closed-loop", "servo_nominal", matrix_seed)]
    canonical_run_id = str(canonical_metrics["run_id"])
    canonical_run_dir = root / canonical_run_id

    completeness_rows = [
        {
            "run_id": canonical_run_id,
            "estimator_mode": canonical_metrics["config"]["estimator_mode"],
            "controller_mode": canonical_metrics["config"]["controller_mode"],
            "bootstrap_control_policy": canonical_metrics["config"]["bootstrap_control_policy"],
            "actuation_preset": canonical_metrics["config"]["actuation_preset"],
            "duration_s": canonical_metrics["timing"]["duration_s"],
            "camera_frames": canonical_metrics["counts"]["camera_frames"],
            "imu_packets": canonical_metrics["counts"]["imu_packets"],
            "commands": canonical_metrics["counts"]["commands"],
            "controller_diagnostics": canonical_metrics["counts"]["controller_diagnostics"],
            "filter_states": canonical_metrics["counts"]["filter_states"],
            "smoother_states": canonical_metrics["counts"]["smoother_states"],
            "uncertainty_states": canonical_metrics["counts"]["uncertainty_states"],
        }
    ]
    matrix_rows = []
    for estimator_mode, controller_mode, actuation_name, seed in matrix_specs:
        metrics = runs[(estimator_mode, controller_mode, actuation_name, seed)]
        matrix_rows.append(
            {
                "label": f"{estimator_mode} / {controller_mode}",
                "run_id": metrics["run_id"],
                "estimator_mode": estimator_mode,
                "controller_mode": controller_mode,
                "mean_position_error_m": metrics["trajectory"]["mean_position_error_m"],
                "p95_position_error_m": metrics["trajectory"]["p95_position_error_m"],
                "mean_waypoint_error_m": metrics["control"]["mean_waypoint_error_m"],
                "completion_fraction": metrics["control"]["completion_fraction"],
                "ik_failure_fraction": metrics["control"]["ik_failure_fraction"],
                "anchor_visible_fraction": metrics["estimation"]["anchor_visible_fraction"],
            }
        )
    reproducibility_rows = []
    for estimator_mode in ("visual", "fused"):
        values = [runs[(estimator_mode, "closed-loop", "servo_nominal", seed)] for seed in reproducibility_seeds]
        position_mean, position_std = _mean_std([value["trajectory"]["mean_position_error_m"] for value in values])
        waypoint_mean, waypoint_std = _mean_std([value["control"]["mean_waypoint_error_m"] for value in values])
        completion_mean, completion_std = _mean_std([value["control"]["completion_fraction"] for value in values])
        ik_mean, ik_std = _mean_std([value["control"]["ik_failure_fraction"] for value in values])
        reproducibility_rows.append(
            {
                "estimator_mode": estimator_mode,
                "controller_mode": "closed-loop",
                "seeds": [int(seed) for seed in reproducibility_seeds],
                "run_ids": [str(value["run_id"]) for value in values],
                "sample_count": len(values),
                "mean_position_error_m": position_mean,
                "std_position_error_m": position_std,
                "mean_waypoint_error_m": waypoint_mean,
                "std_waypoint_error_m": waypoint_std,
                "mean_completion_fraction": completion_mean,
                "std_completion_fraction": completion_std,
                "mean_ik_failure_fraction": ik_mean,
                "std_ik_failure_fraction": ik_std,
            }
        )
    controller_rows = [
        {
            "run_id": canonical_run_id,
            "ik_failure_fraction": canonical_metrics["control"]["ik_failure_fraction"],
            "dominant_safety_reason": canonical_metrics["control"]["dominant_safety_reason"],
            "anchor_visible_fraction": canonical_metrics["estimation"]["anchor_visible_fraction"],
            "anchor_relocalization_count": canonical_metrics["estimation"]["anchor_relocalization_count"],
            "mean_anchor_innovation_norm": canonical_metrics["estimation"]["mean_anchor_innovation_norm"],
            "mean_actuator_tracking_error": canonical_metrics["control"]["mean_actuator_tracking_error"],
        }
    ]
    actuation_rows = []
    for estimator_mode, controller_mode, actuation_name, seed in actuation_specs:
        metrics = runs[(estimator_mode, controller_mode, actuation_name, seed)]
        actuation_rows.append(
            {
                "run_id": metrics["run_id"],
                "actuation_preset": actuation_name,
                "mean_waypoint_error_m": metrics["control"]["mean_waypoint_error_m"],
                "completion_fraction": metrics["control"]["completion_fraction"],
                "ik_failure_fraction": metrics["control"]["ik_failure_fraction"],
                "mean_actuator_tracking_error": metrics["control"]["mean_actuator_tracking_error"],
            }
        )
    uncertainty_rows = [
        {
            "run_id": canonical_run_id,
            "label": "Canonical fused / closed-loop",
            "mean_position_radius_95_m": canonical_metrics["uncertainty_calibration"]["mean_position_radius_95_m"],
            "empirical_95_coverage_percent": canonical_metrics["uncertainty_calibration"]["empirical_95_coverage_percent"],
            "pose_nees": canonical_metrics["uncertainty_calibration"]["pose_nees"],
            "sigma_error_correlation": canonical_metrics["uncertainty_calibration"]["sigma_error_correlation"],
        }
    ]
    residual_rows = [
        {
            "run_id": canonical_run_id,
            "label": "Canonical fused / closed-loop",
            "mean_reprojection_rmse_px": canonical_metrics["residuals"]["mean_reprojection_rmse_px"],
            "p95_reprojection_rmse_px": canonical_metrics["residuals"]["p95_reprojection_rmse_px"],
            "mean_anchor_innovation_norm": canonical_metrics["residuals"]["mean_anchor_innovation_norm"],
            "mean_innovation_norm": canonical_metrics["residuals"]["mean_innovation_norm"],
            "anchor_pnp_success_fraction": canonical_metrics["residuals"]["anchor_pnp_success_fraction"],
            "fallback_only_frame_fraction": canonical_metrics["residuals"]["fallback_only_frame_fraction"],
        }
    ]
    map_quality_rows = [
        {
            "run_id": canonical_run_id,
            "label": "Canonical fused / closed-loop",
            "mean_auxiliary_tag_position_error_m": canonical_metrics["map_quality"]["mean_auxiliary_tag_position_error_m"],
            "p95_auxiliary_tag_position_error_m": canonical_metrics["map_quality"]["p95_auxiliary_tag_position_error_m"],
            "anchor_relocalization_count": canonical_metrics["map_quality"]["anchor_relocalization_count"],
            "anchor_visible_fraction": canonical_metrics["map_quality"]["anchor_visible_fraction"],
        }
    ]

    _write_csv(analysis_dir / "first_pass_runtime_table.csv", completeness_rows)
    _write_csv(analysis_dir / "first_pass_matrix_table.csv", matrix_rows)
    _write_csv(analysis_dir / "first_pass_reproducibility_table.csv", reproducibility_rows)
    _write_csv(analysis_dir / "first_pass_controller_table.csv", controller_rows)
    _write_csv(analysis_dir / "first_pass_actuation_table.csv", actuation_rows)
    _write_csv(analysis_dir / "first_pass_uncertainty_table.csv", uncertainty_rows)
    _write_csv(analysis_dir / "first_pass_residual_table.csv", residual_rows)
    _write_csv(analysis_dir / "first_pass_map_quality_table.csv", map_quality_rows)

    completeness_tex_rows = [
        rf"Canonical run id & {_tex_value(canonical_run_id)} \\",
        rf"Estimator mode & {_tex_value(canonical_metrics['config']['estimator_mode'])} \\",
        rf"Controller mode & {_tex_value(canonical_metrics['config']['controller_mode'])} \\",
        rf"Bootstrap policy & {_tex_value(canonical_metrics['config']['bootstrap_control_policy'])} \\",
        rf"Actuation preset & {_tex_value(canonical_metrics['config']['actuation_preset'])} \\",
        rf"Duration [s] & {_tex_value(canonical_metrics['timing']['duration_s'])} \\",
        rf"Camera frames & {_tex_value(canonical_metrics['counts']['camera_frames'])} \\",
        rf"IMU packets & {_tex_value(canonical_metrics['counts']['imu_packets'])} \\",
        rf"Commands & {_tex_value(canonical_metrics['counts']['commands'])} \\",
        rf"Controller diagnostics & {_tex_value(canonical_metrics['counts']['controller_diagnostics'])} \\",
        rf"Filter states & {_tex_value(canonical_metrics['counts']['filter_states'])} \\",
        rf"Smoother states & {_tex_value(canonical_metrics['counts']['smoother_states'])} \\",
        rf"Uncertainty states & {_tex_value(canonical_metrics['counts']['uncertainty_states'])} \\",
    ]
    matrix_tex_rows = [
        rf"{row['label']} & {_tex_value(row['mean_position_error_m'])} & {_tex_value(row['p95_position_error_m'])} & {_tex_value(row['mean_waypoint_error_m'])} & {_tex_value(row['completion_fraction'], percent=True)} & {_tex_value(row['ik_failure_fraction'], percent=True)} & {_tex_value(row['anchor_visible_fraction'], percent=True)} \\"
        for row in matrix_rows
    ]
    reproducibility_tex_rows = [
        rf"{row['estimator_mode']} / {row['controller_mode']} & {_tex_value(row['sample_count'])} & {_format_mean_std(row['mean_position_error_m'], row['std_position_error_m'])} & {_format_mean_std(row['mean_waypoint_error_m'], row['std_waypoint_error_m'])} & {_format_mean_std(row['mean_completion_fraction'], row['std_completion_fraction'], percent=True)} & {_format_mean_std(row['mean_ik_failure_fraction'], row['std_ik_failure_fraction'], percent=True)} \\"
        for row in reproducibility_rows
    ]
    controller_tex_rows = [
        rf"Canonical fused / closed-loop & {_tex_value(controller_rows[0]['ik_failure_fraction'], percent=True)} & {_tex_value(controller_rows[0]['dominant_safety_reason'])} & {_tex_value(controller_rows[0]['anchor_visible_fraction'], percent=True)} & {_tex_value(controller_rows[0]['anchor_relocalization_count'])} & {_tex_value(controller_rows[0]['mean_anchor_innovation_norm'])} & {_tex_value(controller_rows[0]['mean_actuator_tracking_error'])} \\"
    ]
    actuation_tex_rows = [
        rf"{_tex_value(row['actuation_preset'])} & {_tex_value(row['mean_waypoint_error_m'])} & {_tex_value(row['completion_fraction'], percent=True)} & {_tex_value(row['ik_failure_fraction'], percent=True)} & {_tex_value(row['mean_actuator_tracking_error'])} \\"
        for row in actuation_rows
    ]
    uncertainty_tex_rows = [
        rf"{_tex_value(row['label'])} & {_tex_value(row['mean_position_radius_95_m'])} & {_tex_value(row['empirical_95_coverage_percent'], percent=True)} & {_tex_value(row['pose_nees'])} & {_tex_value(row['sigma_error_correlation'])} \\"
        for row in uncertainty_rows
    ]
    residual_tex_rows = [
        rf"{_tex_value(row['label'])} & {_tex_value(row['mean_reprojection_rmse_px'])} & {_tex_value(row['p95_reprojection_rmse_px'])} & {_tex_value(row['mean_anchor_innovation_norm'])} & {_tex_value(row['mean_innovation_norm'])} & {_tex_value(row['anchor_pnp_success_fraction'], percent=True)} & {_tex_value(row['fallback_only_frame_fraction'], percent=True)} \\"
        for row in residual_rows
    ]
    map_quality_tex_rows = [
        rf"{_tex_value(row['label'])} & {_tex_value(row['mean_auxiliary_tag_position_error_m'])} & {_tex_value(row['p95_auxiliary_tag_position_error_m'])} & {_tex_value(row['anchor_relocalization_count'])} & {_tex_value(row['anchor_visible_fraction'], percent=True)} \\"
        for row in map_quality_rows
    ]

    paper_artifacts_path = report_data_dir / "paper_artifacts.tex"
    paper_artifacts_path.write_text(
        "\n\n".join(
            [
                _macro_definition("IsaacFirstPassCompletenessRows", completeness_tex_rows),
                _macro_definition("IsaacFirstPassMatrixRows", matrix_tex_rows),
                _macro_definition("IsaacFirstPassReproducibilityRows", reproducibility_tex_rows),
                _macro_definition("IsaacFirstPassControllerRows", controller_tex_rows),
                _macro_definition("IsaacFirstPassActuationRows", actuation_tex_rows),
                _macro_definition("IsaacFirstPassUncertaintyRows", uncertainty_tex_rows),
                _macro_definition("IsaacFirstPassResidualRows", residual_tex_rows),
                _macro_definition("IsaacFirstPassMapRows", map_quality_tex_rows),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _copy_if_exists(canonical_run_dir / "analysis" / "report_data" / "trajectory_path.png", report_data_dir / "trajectory_path.png")
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "system_architecture.png",
        report_data_dir / "system_architecture.png",
    )
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "timing_timeline.png",
        report_data_dir / "timing_timeline.png",
    )
    _copy_if_exists(canonical_run_dir / "analysis" / "report_data" / "path_tracking.png", report_data_dir / "path_tracking.png")
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "uncertainty_calibration.png",
        report_data_dir / "uncertainty_calibration.png",
    )
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "residual_histogram.png",
        report_data_dir / "residual_histogram.png",
    )
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "actuator_command_vs_realized.png",
        report_data_dir / "actuator_command_vs_realized.png",
    )
    _copy_if_exists(
        canonical_run_dir / "analysis" / "report_data" / "smoother_convergence.png",
        report_data_dir / "smoother_convergence.png",
    )
    _controller_diagnostic_figure(
        canonical_run_dir,
        report_data_dir / "ik_failure_timeline.png",
        canonical_metrics["control"],
    )

    placeholders_remaining = r"\ArtifactPending{}" in paper_artifacts_path.read_text(encoding="utf-8")
    summary = {
        "schema_version": 2,
        "canonical_run_id": canonical_run_id,
        "matrix_runs": matrix_rows,
        "reproducibility": reproducibility_rows,
        "actuation": actuation_rows,
        "controller": controller_rows,
        "uncertainty": uncertainty_rows,
        "residuals": residual_rows,
        "map_quality": map_quality_rows,
        "publication": {
            "artifact_source": str(artifact_source),
            "suite_dir": str(suite_dir),
            "report_data_dir": str(report_data_dir),
            "paper_artifacts_tex": str(paper_artifacts_path),
            "placeholders_remaining": bool(placeholders_remaining),
        },
    }
    (analysis_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "suite_dir": str(suite_dir),
        "canonical_run_id": canonical_run_id,
        "summary_json": str(analysis_dir / "suite_summary.json"),
        "paper_artifacts_tex": str(paper_artifacts_path),
    }


__all__ = ["generate_first_pass_suite_artifacts"]
