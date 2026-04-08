"""Figure generation for Isaac reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _vector_or_none(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _position_from_gt_row(row: dict[str, Any]) -> np.ndarray | None:
    required_keys = ("px", "py", "pz")
    if all(key in row for key in required_keys):
        values = [_float_or_none(row[key]) for key in required_keys]
        if all(value is not None for value in values):
            return np.asarray(values, dtype=np.float64)
    return None


def _save_text_figure(path: Path, *, title: str, lines: list[str]) -> None:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 26, 32), 2, cv2.LINE_AA)
    y = 90
    for line in lines:
        cv2.putText(canvas, line[:110], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (62, 66, 72), 1, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(path), canvas)


def _plot_canvas(title: str, subtitle: str | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    left, right, top, bottom = 90, 40, 60, 70
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (24, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    origin_x = left
    origin_y = 420 - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (1080 - right, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    return canvas, (left, right, top, bottom)


def _save_line_plot(path: Path, *, title: str, x_values: list[float], y_values: list[float], y_label: str) -> None:
    if not x_values or not y_values or len(x_values) != len(y_values):
        _save_text_figure(path, title=title, lines=["No data available for this figure."])
        return
    xs = np.asarray(x_values, dtype=np.float64)
    ys = np.asarray(y_values, dtype=np.float64)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        _save_text_figure(path, title=title, lines=["No finite samples available."])
        return
    xs = xs[finite]
    ys = ys[finite]
    canvas, (left, right, top, bottom) = _plot_canvas(title, y_label)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0
    points = []
    for x, y in zip(xs, ys):
        x_px = origin_x + int(round((x - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((y - min_y) / (max_y - min_y) * plot_height))
        points.append((x_px, y_px))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (53, 102, 188), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_xy_plot(
    path: Path,
    *,
    title: str,
    tracks: list[tuple[list[float], list[float], tuple[int, int, int]]],
    point_sets: list[tuple[list[tuple[float, float]], tuple[int, int, int]]] | None = None,
    footer: str,
) -> None:
    valid_tracks = [(xs, ys, color) for xs, ys, color in tracks if xs and ys and len(xs) == len(ys)]
    valid_points = point_sets or []
    all_x = [value for xs, _, _ in valid_tracks for value in xs] + [x for points, _ in valid_points for x, _ in points]
    all_y = [value for _, ys, _ in valid_tracks for value in ys] + [y for points, _ in valid_points for _, y in points]
    if not all_x or not all_y:
        _save_text_figure(path, title=title, lines=["No planar trajectory data available."])
        return

    canvas, (left, right, top, bottom) = _plot_canvas(title, footer)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    min_x, max_x = float(min(all_x)), float(max(all_x))
    min_y, max_y = float(min(all_y)), float(max(all_y))
    if abs(max_x - min_x) < 1e-12:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-12:
        max_y = min_y + 1.0

    def project(x_value: float, y_value: float) -> tuple[int, int]:
        x_px = origin_x + int(round((x_value - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((y_value - min_y) / (max_y - min_y) * plot_height))
        return x_px, y_px

    for xs, ys, color in valid_tracks:
        points = [project(x_value, y_value) for x_value, y_value in zip(xs, ys)]
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
    for points, color in valid_points:
        for x_value, y_value in points:
            cv2.circle(canvas, project(x_value, y_value), 5, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_histogram(path: Path, *, title: str, values: list[float], x_label: str) -> None:
    if not values:
        _save_text_figure(path, title=title, lines=["No histogram samples available."])
        return
    samples = np.asarray(values, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        _save_text_figure(path, title=title, lines=["No finite histogram samples available."])
        return
    counts, edges = np.histogram(finite, bins=min(12, max(4, finite.size)))
    canvas, (left, right, top, bottom) = _plot_canvas(title, x_label)
    origin_x = left
    origin_y = 420 - bottom
    plot_width = 1080 - left - right
    plot_height = 420 - top - bottom
    max_count = max(int(np.max(counts)), 1)
    bin_width = plot_width / len(counts)
    for index, count in enumerate(counts):
        x0 = int(origin_x + index * bin_width)
        x1 = int(origin_x + (index + 1) * bin_width - 4)
        y1 = origin_y
        y0 = origin_y - int(round(count / max_count * plot_height))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (53, 102, 188), -1)
    cv2.imwrite(str(path), canvas)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _first_realized_position(row: dict[str, Any]) -> float | None:
    raw = row.get("positions")
    if raw in ("", None):
        return None
    return _float_or_none(str(raw).split("|")[0])


def _first_command_value(row: dict[str, Any]) -> float | None:
    raw_value = _float_or_none(row.get("command_value"))
    if raw_value is not None:
        return raw_value
    for key in ("desired_positions", "effective_positions"):
        raw = row.get(key)
        if raw in ("", None):
            continue
        value = _float_or_none(str(raw).split("|")[0])
        if value is not None:
            return value
    return None


def _matched_position_error_series(
    filter_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    uncertainty_rows: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    gt_samples: list[tuple[float, np.ndarray]] = []
    for row in gt_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _position_from_gt_row(row)
        if timestamp is not None and position is not None:
            gt_samples.append((timestamp, position))
    if not gt_samples:
        return [], [], []
    gt_times = np.asarray([item[0] for item in gt_samples], dtype=np.float64)
    radii = {
        float(row["timestamp_s"]): float(row["position_radius_95_m"])
        for row in uncertainty_rows
        if "timestamp_s" in row and "position_radius_95_m" in row
    }
    times: list[float] = []
    errors: list[float] = []
    radius_values: list[float] = []
    for row in filter_rows:
        timestamp = _float_or_none(row.get("timestamp_s"))
        position = _vector_or_none(row.get("position_world_m"))
        if timestamp is None or position is None:
            continue
        gt_index = int(np.argmin(np.abs(gt_times - timestamp)))
        error = float(np.linalg.norm(position - gt_samples[gt_index][1]))
        radius = radii.get(float(timestamp))
        if radius is None:
            continue
        times.append(float(timestamp))
        errors.append(error)
        radius_values.append(float(radius))
    return times, errors, radius_values


def write_isaac_report_figures(run_dir: str | Path, metrics: dict[str, Any]) -> dict[str, str]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    report_data_dir = analysis_dir / "report_data"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_data_dir.mkdir(parents=True, exist_ok=True)

    filter_rows = _read_jsonl_rows(resolved / "estimates" / "filter_state.jsonl")
    uncertainty_rows = _read_jsonl_rows(resolved / "estimates" / "uncertainty.jsonl")
    command_rows = _read_csv_rows(resolved / "raw" / "commands.csv")
    realized_rows = _read_csv_rows(resolved / "raw" / "realized_joints.csv")
    camera_gt_rows = _read_csv_rows(resolved / "gt" / "camera_gt.csv")

    metrics_figure = analysis_dir / "isaac_metrics_summary.png"
    uncertainty_figure = analysis_dir / "isaac_uncertainty_timeline.png"
    command_figure = analysis_dir / "isaac_command_timeline.png"

    _save_text_figure(
        metrics_figure,
        title="Isaac Run Summary",
        lines=[
            f"run id: {metrics['run_id']}",
            f"stage: {metrics['manifest']['stage_usd_path']}",
            f"robot preset: {metrics['manifest']['robot_preset']}",
            f"camera frames: {metrics['counts']['camera_frames']}",
            f"imu packets: {metrics['counts']['imu_packets']}",
            f"aux tags: {metrics['counts']['unique_detected_auxiliary_tags']}",
        ],
    )
    _save_line_plot(
        uncertainty_figure,
        title="Position Radius 95% Timeline",
        x_values=[float(row["timestamp_s"]) for row in uncertainty_rows if "timestamp_s" in row],
        y_values=[float(row["position_radius_95_m"]) for row in uncertainty_rows if "position_radius_95_m" in row],
        y_label="m",
    )
    _save_line_plot(
        command_figure,
        title="Command Magnitude Timeline",
        x_values=[float(row["timestamp_s"]) for row in command_rows if "timestamp_s" in row],
        y_values=[abs(value) for row in command_rows if (value := _first_command_value(row)) is not None],
        y_label="abs command",
    )

    filter_positions = [(_float_or_none(row.get("timestamp_s")), _vector_or_none(row.get("position_world_m"))) for row in filter_rows]
    trajectory_times = [timestamp for timestamp, position in filter_positions if timestamp is not None and position is not None]
    trajectory_x = [float(position[0]) for timestamp, position in filter_positions if timestamp is not None and position is not None]
    trajectory_y = [float(position[1]) for timestamp, position in filter_positions if timestamp is not None and position is not None]
    waypoint_points = []
    for waypoint in metrics["manifest"]["controller_config"].get("waypoints", []):
        position = _vector_or_none(waypoint.get("position_world_m")) if isinstance(waypoint, dict) else None
        if position is not None:
            waypoint_points.append((float(position[0]), float(position[1])))

    architecture_path = report_data_dir / "system_architecture.png"
    timing_path = report_data_dir / "timing_timeline.png"
    trajectory_path = report_data_dir / "trajectory_path.png"
    tracking_path = report_data_dir / "path_tracking.png"
    convergence_path = report_data_dir / "smoother_convergence.png"
    residual_histogram_path = report_data_dir / "residual_histogram.png"
    calibration_path = report_data_dir / "uncertainty_calibration.png"
    actuator_path = report_data_dir / "actuator_command_vs_realized.png"

    _save_text_figure(
        architecture_path,
        title="Anchored VIO Runtime Architecture",
        lines=[
            "Isaac runtime -> raw camera / IMU / command logs",
            "AprilTag frontend -> anchored filter -> fixed-lag smoother",
            "Controller closes loop on anchored-frame path estimate",
            f"ROS 2 bridge enabled: {metrics['manifest']['ros2_bridge_used']}",
        ],
    )
    _save_text_figure(
        timing_path,
        title="Sensor And Estimator Timing Summary",
        lines=[
            f"physics rate [Hz]: {metrics['config']['physics_rate_hz']}",
            f"imu rate [Hz]: {metrics['config']['imu_rate_hz']}",
            f"camera rate [Hz]: {metrics['config']['camera_rate_hz']}",
            f"filter rate [Hz]: {metrics['config']['filter_rate_hz']}",
            f"smoother rate [Hz]: {metrics['config']['smoother_rate_hz']}",
            f"controller rate [Hz]: {metrics['config']['controller_rate_hz']}",
        ],
    )
    _save_xy_plot(
        trajectory_path,
        title="Anchored-Frame Trajectory",
        tracks=[(trajectory_x, trajectory_y, (53, 102, 188))],
        point_sets=[(waypoint_points, (188, 92, 60))],
        footer="x-y projection in anchored frame",
    )
    _save_xy_plot(
        tracking_path,
        title="Path Tracking In Anchored Frame",
        tracks=[(trajectory_x, trajectory_y, (31, 160, 92))],
        point_sets=[(waypoint_points, (188, 92, 60))],
        footer=f"completion fraction: {metrics['control']['completion_fraction']}",
    )
    _save_line_plot(
        convergence_path,
        title="Smoother / Uncertainty Convergence",
        x_values=[float(row["timestamp_s"]) for row in uncertainty_rows if "timestamp_s" in row],
        y_values=[float(row["position_radius_95_m"]) for row in uncertainty_rows if "position_radius_95_m" in row],
        y_label="position radius 95% [m]",
    )
    _save_histogram(
        residual_histogram_path,
        title="Innovation Histogram",
        values=[
            float(row.get("innovation_diagnostics", {}).get("last_innovation_norm", 0.0))
            for row in filter_rows
            if isinstance(row.get("innovation_diagnostics"), dict)
        ],
        x_label="innovation norm",
    )
    error_times, position_errors, radius_values = _matched_position_error_series(filter_rows, camera_gt_rows, uncertainty_rows)
    if error_times:
        _save_xy_plot(
            calibration_path,
            title="Uncertainty Calibration",
            tracks=[
                (error_times, position_errors, (188, 92, 60)),
                (error_times, radius_values, (53, 102, 188)),
            ],
            footer="orange = error, blue = reported 95% radius",
        )
    else:
        _save_text_figure(
            calibration_path,
            title="Uncertainty Calibration",
            lines=["Ground-truth pose samples are required to draw the coverage plot."],
        )
    realized_times = [float(row["timestamp_s"]) for row in realized_rows if _float_or_none(row.get("timestamp_s")) is not None and _first_realized_position(row) is not None]
    realized_values = [float(_first_realized_position(row)) for row in realized_rows if _float_or_none(row.get("timestamp_s")) is not None and _first_realized_position(row) is not None]
    command_times = [
        float(row["timestamp_s"])
        for row in command_rows
        if _float_or_none(row.get("timestamp_s")) is not None and _first_command_value(row) is not None
    ]
    command_values = [
        float(_first_command_value(row))
        for row in command_rows
        if _float_or_none(row.get("timestamp_s")) is not None and _first_command_value(row) is not None
    ]
    _save_xy_plot(
        actuator_path,
        title="Actuator Command Versus Realized Motion",
        tracks=[
            (command_times, command_values, (53, 102, 188)),
            (realized_times, realized_values, (188, 92, 60)),
        ],
        footer="time on x-axis, first joint signal on y-axis",
    )
    return {
        "metrics_summary": str(metrics_figure),
        "uncertainty_timeline": str(uncertainty_figure),
        "command_timeline": str(command_figure),
        "system_architecture": str(architecture_path),
        "timing_timeline": str(timing_path),
        "trajectory_path": str(trajectory_path),
        "path_tracking": str(tracking_path),
        "smoother_convergence": str(convergence_path),
        "residual_histogram": str(residual_histogram_path),
        "uncertainty_calibration": str(calibration_path),
        "actuator_command_vs_realized": str(actuator_path),
    }


def write_isaac_estimator_quality_figures(run_dir: str | Path, quality: dict[str, Any]) -> dict[str, str]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(quality.get("summary", {}))

    anchor_vs_aux_path = analysis_dir / "anchor_vs_aux_residuals.png"
    smoother_timeline_path = analysis_dir / "smoother_correction_timeline.png"

    _save_text_figure(
        anchor_vs_aux_path,
        title="Anchor vs Auxiliary Residual Quality",
        lines=[
            f"anchor mean / p95 [px]: {summary.get('anchor_mean_reprojection_rmse_px')} / {summary.get('anchor_p95_reprojection_rmse_px')}",
            f"aux mean / p95 [px]: {summary.get('auxiliary_mean_reprojection_rmse_px')} / {summary.get('auxiliary_p95_reprojection_rmse_px')}",
            f"native mean / p95 [px]: {summary.get('native_mean_reprojection_rmse_px')} / {summary.get('native_p95_reprojection_rmse_px')}",
            f"fallback mean / p95 [px]: {summary.get('fallback_mean_reprojection_rmse_px')} / {summary.get('fallback_p95_reprojection_rmse_px')}",
            f"pre-relocalization mean / p95 [px]: {summary.get('pre_relocalization_mean_reprojection_rmse_px')} / {summary.get('pre_relocalization_p95_reprojection_rmse_px')}",
            f"post-relocalization mean / p95 [px]: {summary.get('post_relocalization_mean_reprojection_rmse_px')} / {summary.get('post_relocalization_p95_reprojection_rmse_px')}",
            f"accepted / rejected aux updates: {summary.get('accepted_auxiliary_updates')} / {summary.get('rejected_auxiliary_updates')}",
        ],
    )
    smoother_rows = list(quality.get("smoother_timeline", []))
    _save_line_plot(
        smoother_timeline_path,
        title="Smoother Correction Timeline",
        x_values=[
            float(row["timestamp_s"])
            for row in smoother_rows
            if _float_or_none(row.get("timestamp_s")) is not None and row.get("feedback_correction_norm_m") is not None
        ],
        y_values=[
            float(row["feedback_correction_norm_m"])
            for row in smoother_rows
            if _float_or_none(row.get("timestamp_s")) is not None and row.get("feedback_correction_norm_m") is not None
        ],
        y_label="feedback correction norm [m]",
    )
    return {
        "anchor_vs_aux_residuals": str(anchor_vs_aux_path),
        "smoother_correction_timeline": str(smoother_timeline_path),
    }
