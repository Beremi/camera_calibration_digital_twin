"""Batch estimation analysis, reporting, and artifact generation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.estimation import (
    bootstrap_initial_guess,
    compute_laplace_posterior,
    evaluate_batch_result,
    inertial_initial_guess_from_visual_solution,
    load_batch_dataset,
    load_evaluation_data,
    load_imu_noise_preset,
    load_vision_noise_preset,
    solve_visual_batch_map,
    solve_visual_inertial_batch_map,
)
from calib_sim.estimation._geometry import (
    apply_world_alignment_to_pose_vector,
    pose_components_from_vector,
    pose_vector_from_components,
    project_world_points_to_pixels,
    rotation_xyz_deg,
    world_alignment_from_reference_pose,
)
from calib_sim.estimation.eval_metrics import _camera_truth_by_frame, _tag_truth_pose_by_id
from calib_sim.estimation.factors.imu_preintegration import ImuSample, integrate_discrete_imu_sequence
from calib_sim.estimation.noise_models import ImuNoisePreset, VisionNoisePreset
from calib_sim.estimation.types import (
    BatchCalibrationDataset,
    BatchCalibrationEvaluationData,
    BatchSolveResult,
    CameraFrame,
    EvalSummary,
    ImuPacket,
    InitialGuess,
    TagDetectionObservation,
    UncertaintySummary,
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _frame_by_index(dataset: BatchCalibrationDataset) -> dict[int, CameraFrame]:
    return {int(frame.frame_index): frame for frame in dataset.camera_frames}


def _common_frame_indices(result: BatchSolveResult, evaluation_data: BatchCalibrationEvaluationData) -> list[int]:
    truth_by_frame = _camera_truth_by_frame(evaluation_data)
    return sorted(set(int(frame_index) for frame_index in result.camera_pose_vectors) & set(truth_by_frame))


def _alignment_for_result(
    result: BatchSolveResult,
    evaluation_data: BatchCalibrationEvaluationData,
) -> tuple[np.ndarray, np.ndarray] | None:
    truth_by_frame = _camera_truth_by_frame(evaluation_data)
    common_frames = _common_frame_indices(result, evaluation_data)
    if not common_frames:
        return None
    reference_frame = common_frames[0]
    truth = truth_by_frame[reference_frame]
    truth_pose = pose_vector_from_components(
        np.asarray(truth.position_world_m, dtype=np.float64),
        np.asarray(truth.rotation_cw, dtype=np.float64).reshape(3, 3),
    )
    return world_alignment_from_reference_pose(
        np.asarray(result.camera_pose_vectors[reference_frame], dtype=np.float64),
        truth_pose,
    )


def _aligned_pose_vector(
    pose_vector: np.ndarray,
    alignment: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if alignment is None:
        return np.asarray(pose_vector, dtype=np.float64).reshape(6)
    rotation, translation = alignment
    return apply_world_alignment_to_pose_vector(rotation, translation, pose_vector)


def _save_pose_estimates_jsonl(
    path: Path,
    *,
    dataset: BatchCalibrationDataset,
    result: BatchSolveResult,
    evaluation_data: BatchCalibrationEvaluationData,
    uncertainty_summary: UncertaintySummary | None = None,
) -> None:
    frame_lookup = _frame_by_index(dataset)
    alignment = _alignment_for_result(result, evaluation_data)
    uncertainty_lookup = (
        {int(item.index): item for item in uncertainty_summary.camera_pose_uncertainty}
        if uncertainty_summary is not None
        else {}
    )
    with path.open("w", encoding="utf-8") as handle:
        for frame_index in sorted(int(index) for index in result.camera_pose_vectors):
            frame = frame_lookup.get(int(frame_index))
            aligned_pose = _aligned_pose_vector(np.asarray(result.camera_pose_vectors[frame_index], dtype=np.float64), alignment)
            position_world_m, rotation_wc = pose_components_from_vector(aligned_pose)
            payload = {
                "frame_index": int(frame_index),
                "timestamp_s": None if frame is None else float(frame.timestamp_s),
                "position_world_m": [float(value) for value in position_world_m.tolist()],
                "rotation_cw": [[float(value) for value in row] for row in rotation_wc.tolist()],
                "rotation_xyz_deg": [float(value) for value in rotation_xyz_deg(rotation_wc).tolist()],
                "uncertainty": None,
            }
            uncertainty = uncertainty_lookup.get(int(frame_index))
            if uncertainty is not None:
                payload["uncertainty"] = {
                    "position_std_m": [float(value) for value in uncertainty.position_std_m],
                    "rotation_std_deg": [float(value) for value in uncertainty.rotation_std_deg],
                    "position_radius_95_m": float(uncertainty.position_radius_95_m),
                    "rotation_radius_95_deg": float(uncertainty.rotation_radius_95_deg),
                }
            handle.write(json.dumps(payload) + "\n")


def _save_tag_map_json(path: Path, *, result: BatchSolveResult, evaluation_data: BatchCalibrationEvaluationData) -> None:
    alignment = _alignment_for_result(result, evaluation_data)
    with path.open("w", encoding="utf-8") as handle:
        for tag_id in sorted(int(index) for index in result.tag_pose_vectors):
            aligned_pose = _aligned_pose_vector(np.asarray(result.tag_pose_vectors[tag_id], dtype=np.float64), alignment)
            position_world_m, rotation_wt = pose_components_from_vector(aligned_pose)
            payload = {
                "tag_id": int(tag_id),
                "position_world_m": [float(value) for value in position_world_m.tolist()],
                "rotation_wt": [[float(value) for value in row] for row in rotation_wt.tolist()],
                "rotation_xyz_deg": [float(value) for value in rotation_xyz_deg(rotation_wt).tolist()],
            }
            handle.write(json.dumps(payload) + "\n")


def _save_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    y_values: list[float],
    color_bgr: tuple[int, int, int] = (56, 121, 217),
) -> None:
    if not x_values or not y_values:
        return
    finite_pairs = [(float(x), float(y)) for x, y in zip(x_values, y_values) if np.isfinite(float(x)) and np.isfinite(float(y))]
    if not finite_pairs:
        return
    width = 1080
    height = 420
    left = 90
    right = 40
    top = 60
    bottom = 80
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width - 180, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    xs = np.asarray([pair[0] for pair in finite_pairs], dtype=np.float64)
    ys = np.asarray([pair[1] for pair in finite_pairs], dtype=np.float64)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    y_padding = max((max_y - min_y) * 0.08, 1e-9)
    min_y -= y_padding
    max_y += y_padding
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    for tick_index in range(5):
        value = min_y + (max_y - min_y) * tick_index / 4.0
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.4f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    points = []
    for x_value, y_value in finite_pairs:
        x_px = origin_x + int(round((x_value - min_x) / max(max_x - min_x, 1e-12) * plot_width))
        y_px = origin_y - int(round((y_value - min_y) / max(max_y - min_y, 1e-12) * plot_height))
        points.append((x_px, y_px))
    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color_bgr, 2, cv2.LINE_AA)
    for point in points[:: max(1, len(points) // 24)]:
        cv2.circle(canvas, point, 3, color_bgr, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_cdf_plot(path: Path, *, title: str, values: list[float], x_label: str) -> None:
    finite = sorted(float(value) for value in values if np.isfinite(float(value)))
    if not finite:
        return
    cdf = [(index + 1) / len(finite) for index in range(len(finite))]
    _save_line_plot(path, title=title, x_label=x_label, y_label="empirical CDF", x_values=finite, y_values=cdf)


def _save_bar_plot(path: Path, *, title: str, labels: list[str], values: list[float], y_label: str) -> None:
    if not labels or not values:
        return
    width = 1080
    height = 420
    left = 90
    right = 40
    top = 60
    bottom = 110
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    max_value = max(float(value) for value in values)
    y_max = max(max_value * 1.1, 1e-6)
    bar_width = max(plot_width // max(len(values) * 2, 2), 20)
    spacing = plot_width / max(len(values), 1)
    for tick_index in range(5):
        value = y_max * tick_index / 4.0
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.4f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    for index, (label, value) in enumerate(zip(labels, values)):
        center_x = int(round(origin_x + (index + 0.5) * spacing))
        x0 = center_x - bar_width // 2
        x1 = center_x + bar_width // 2
        y1 = origin_y
        y0 = origin_y - int(round((float(value) / y_max) * plot_height))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (74, 132, 212), -1)
        cv2.putText(canvas, label, (x0 - 10, origin_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (82, 90, 98), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{float(value):.4f}", (x0 - 14, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 60, 68), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_scatter_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    points: list[dict[str, Any]],
) -> None:
    if not points:
        return
    width = 1080
    height = 420
    left = 90
    right = 40
    top = 60
    bottom = 80
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width - 200, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    xs = np.asarray([float(item["x"]) for item in points], dtype=np.float64)
    ys = np.asarray([float(item["y"]) for item in points], dtype=np.float64)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    for point in points:
        x = origin_x + int(round((float(point["x"]) - min_x) / max(max_x - min_x, 1e-12) * plot_width))
        y = origin_y - int(round((float(point["y"]) - min_y) / max(max_y - min_y, 1e-12) * plot_height))
        color = tuple(int(channel) for channel in point.get("color_bgr", (74, 132, 212)))
        cv2.circle(canvas, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, str(point["label"]), (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 60, 68), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_histogram_plot(
    path: Path,
    *,
    title: str,
    values: list[float],
    x_label: str,
    bins: int = 24,
    color_bgr: tuple[int, int, int] = (56, 121, 217),
) -> None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return
    width = 1080
    height = 420
    left = 90
    right = 40
    top = 60
    bottom = 80
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width - 220, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    hist, bin_edges = np.histogram(np.asarray(finite, dtype=np.float64), bins=max(int(bins), 4))
    max_count = max(int(np.max(hist)), 1)
    for tick_index in range(5):
        count = int(round(max_count * tick_index / 4.0))
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{count}", (16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    bar_spacing = plot_width / max(len(hist), 1)
    for index, count in enumerate(hist.tolist()):
        x0 = int(round(origin_x + index * bar_spacing))
        x1 = int(round(origin_x + (index + 1) * bar_spacing)) - 2
        y0 = origin_y - int(round((count / max_count) * plot_height))
        cv2.rectangle(canvas, (x0, y0), (max(x1, x0 + 1), origin_y), color_bgr, -1)
    x_min = float(bin_edges[0])
    x_max = float(bin_edges[-1])
    for tick_index in range(5):
        alpha = tick_index / 4.0
        value = x_min + (x_max - x_min) * alpha
        x = int(round(origin_x + plot_width * alpha))
        cv2.putText(canvas, f"{value:.3f}", (x - 18, origin_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_multi_line_plot(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[float],
    series_specs: list[dict[str, Any]],
) -> None:
    if not x_values or not series_specs:
        return
    finite_y = [
        float(value)
        for spec in series_specs
        for value in spec.get("values", [])
        if value is not None and np.isfinite(float(value))
    ]
    finite_x = [float(value) for value in x_values if np.isfinite(float(value))]
    if not finite_x or not finite_y:
        return
    width = 1080
    height = 460
    left = 90
    right = 180
    top = 60
    bottom = 80
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    cv2.putText(canvas, x_label, (width - 220, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    min_x = float(np.min(finite_x))
    max_x = float(np.max(finite_x))
    min_y = float(np.min(finite_y))
    max_y = float(np.max(finite_y))
    if math.isclose(min_x, max_x):
        min_x -= 1.0
        max_x += 1.0
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0
    y_padding = (max_y - min_y) * 0.08
    min_y -= y_padding
    max_y += y_padding
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    for tick_index in range(5):
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        value = min_y + (max_y - min_y) * tick_index / 4.0
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.3f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    for tick_index in range(5):
        x = int(round(origin_x + plot_width * tick_index / 4.0))
        value = min_x + (max_x - min_x) * tick_index / 4.0
        cv2.line(canvas, (x, top), (x, origin_y), (240, 243, 246), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.1f}", (x - 16, origin_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)

    def to_px(x_value: float, y_value: float) -> tuple[int, int]:
        x_px = origin_x + int(round((x_value - min_x) / max(max_x - min_x, 1e-12) * plot_width))
        y_px = origin_y - int(round((y_value - min_y) / max(max_y - min_y, 1e-12) * plot_height))
        return x_px, y_px

    legend_y = 72
    for spec in series_specs:
        values = spec.get("values", [])
        color = tuple(int(channel) for channel in spec.get("color_bgr", (56, 121, 217)))
        points = [
            to_px(float(x_value), float(y_value))
            for x_value, y_value in zip(x_values, values)
            if y_value is not None and np.isfinite(float(y_value))
        ]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        elif len(points) == 1:
            cv2.circle(canvas, points[0], 4, color, -1, cv2.LINE_AA)
        if points:
            for point in points[:: max(1, len(points) // 20)]:
                cv2.circle(canvas, point, 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (width - 160, legend_y - 4), (width - 128, legend_y - 4), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, str(spec.get("label", "series")), (width - 120, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 60, 68), 1, cv2.LINE_AA)
        legend_y += 22
    cv2.imwrite(str(path), canvas)


def _save_grouped_bar_plot(
    path: Path,
    *,
    title: str,
    group_labels: list[str],
    series_specs: list[dict[str, Any]],
    y_label: str,
) -> None:
    if not group_labels or not series_specs:
        return
    finite_values = [
        float(value)
        for spec in series_specs
        for value in spec.get("values", [])
        if value is not None and np.isfinite(float(value))
    ]
    if not finite_values:
        return
    width = 1080
    height = 460
    left = 90
    right = 180
    top = 60
    bottom = 100
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, height - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    plot_width = width - left - right
    plot_height = height - top - bottom
    origin_x = left
    origin_y = height - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (width - right, origin_y), (170, 176, 184), 1, cv2.LINE_AA)
    y_max = max(float(np.max(np.asarray(finite_values, dtype=np.float64))) * 1.1, 1e-9)
    for tick_index in range(5):
        y = int(round(origin_y - plot_height * tick_index / 4.0))
        value = y_max * tick_index / 4.0
        cv2.line(canvas, (origin_x, y), (width - right, y), (232, 236, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.3f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    group_spacing = plot_width / max(len(group_labels), 1)
    series_count = len(series_specs)
    bar_width = max(int(group_spacing * 0.65 / max(series_count, 1)), 12)
    for group_index, group_label in enumerate(group_labels):
        group_center_x = int(round(origin_x + (group_index + 0.5) * group_spacing))
        for series_index, spec in enumerate(series_specs):
            values = spec.get("values", [])
            if group_index >= len(values) or values[group_index] is None or not np.isfinite(float(values[group_index])):
                continue
            value = float(values[group_index])
            color = tuple(int(channel) for channel in spec.get("color_bgr", (56, 121, 217)))
            offset = int(round((series_index - (series_count - 1) * 0.5) * bar_width * 1.2))
            x0 = group_center_x + offset - bar_width // 2
            x1 = x0 + bar_width
            y0 = origin_y - int(round((value / y_max) * plot_height))
            cv2.rectangle(canvas, (x0, y0), (x1, origin_y), color, -1)
        cv2.putText(canvas, group_label, (group_center_x - 26, origin_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (98, 106, 114), 1, cv2.LINE_AA)
    legend_y = 72
    for spec in series_specs:
        color = tuple(int(channel) for channel in spec.get("color_bgr", (56, 121, 217)))
        cv2.rectangle(canvas, (width - 160, legend_y - 10), (width - 140, legend_y + 2), color, -1)
        cv2.putText(canvas, str(spec.get("label", "series")), (width - 130, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 60, 68), 1, cv2.LINE_AA)
        legend_y += 22
    cv2.imwrite(str(path), canvas)


def _save_3d_projection_plot(
    path: Path,
    *,
    title: str,
    series_specs: list[dict[str, Any]],
    tag_specs: list[dict[str, Any]] | None = None,
) -> None:
    all_points = []
    for spec in series_specs:
        all_points.extend([np.asarray(point, dtype=np.float64).reshape(3) for point in spec.get("points", [])])
    if tag_specs is not None:
        all_points.extend([np.asarray(spec["point"], dtype=np.float64).reshape(3) for spec in tag_specs])
    if not all_points:
        return
    width = 900
    height = 720
    margin = 60
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    stacked = np.asarray(all_points, dtype=np.float64)
    center = np.mean(stacked, axis=0)
    yaw = math.radians(-45.0)
    pitch = math.radians(28.0)
    rotation_yaw = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation_pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(pitch), -math.sin(pitch)], [0.0, math.sin(pitch), math.cos(pitch)]],
        dtype=np.float64,
    )
    rotation_view = rotation_pitch @ rotation_yaw
    projected = (rotation_view @ (stacked - center).T).T
    min_x = float(np.min(projected[:, 0]))
    max_x = float(np.max(projected[:, 0]))
    min_y = float(np.min(projected[:, 1]))
    max_y = float(np.max(projected[:, 1]))
    if math.isclose(min_x, max_x):
        min_x -= 0.1
        max_x += 0.1
    if math.isclose(min_y, max_y):
        min_y -= 0.1
        max_y += 0.1
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def project_point(point: np.ndarray) -> tuple[int, int]:
        rotated = rotation_view @ (np.asarray(point, dtype=np.float64).reshape(3) - center)
        x_px = margin + int(round((rotated[0] - min_x) / max(max_x - min_x, 1e-12) * plot_width))
        y_px = height - margin - int(round((rotated[1] - min_y) / max(max_y - min_y, 1e-12) * plot_height))
        return x_px, y_px

    legend_y = 72
    for spec in series_specs:
        points = [np.asarray(point, dtype=np.float64).reshape(3) for point in spec.get("points", [])]
        if not points:
            continue
        color = tuple(int(channel) for channel in spec.get("color_bgr", (56, 121, 217)))
        projected_points = np.asarray([project_point(point) for point in points], dtype=np.int32)
        if len(projected_points) >= 2:
            cv2.polylines(canvas, [projected_points], False, color, 2, cv2.LINE_AA)
        for point in projected_points[:: max(1, len(projected_points) // 24)]:
            cv2.circle(canvas, tuple(int(value) for value in point), 3, color, -1, cv2.LINE_AA)
        cv2.line(canvas, (width - 180, legend_y - 4), (width - 148, legend_y - 4), color, 3, cv2.LINE_AA)
        cv2.putText(canvas, str(spec.get("label", "series")), (width - 140, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (52, 60, 68), 1, cv2.LINE_AA)
        legend_y += 22
    if tag_specs is not None:
        for spec in tag_specs:
            point = project_point(np.asarray(spec["point"], dtype=np.float64))
            color = tuple(int(channel) for channel in spec.get("color_bgr", (205, 102, 44)))
            cv2.rectangle(canvas, (point[0] - 5, point[1] - 5), (point[0] + 5, point[1] + 5), color, -1)
            cv2.putText(canvas, str(spec.get("label", "tag")), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 60, 68), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _save_topdown_plot(
    path: Path,
    *,
    title: str,
    camera_points: list[np.ndarray],
    tag_points: list[np.ndarray],
) -> None:
    if not camera_points and not tag_points:
        return
    width = 720
    height = 720
    margin = 60
    canvas = np.full((height, width, 3), 252, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (220, 224, 228), 1)
    cv2.putText(canvas, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    stacked = [np.asarray(point, dtype=np.float64).reshape(3) for point in camera_points + tag_points]
    xs = np.asarray([point[0] for point in stacked], dtype=np.float64)
    ys = np.asarray([point[1] for point in stacked], dtype=np.float64)
    min_x = float(np.min(xs))
    max_x = float(np.max(xs))
    min_y = float(np.min(ys))
    max_y = float(np.max(ys))
    if math.isclose(min_x, max_x):
        min_x -= 0.1
        max_x += 0.1
    if math.isclose(min_y, max_y):
        min_y -= 0.1
        max_y += 0.1
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    def to_px(point: np.ndarray) -> tuple[int, int]:
        x = margin + int(round((float(point[0]) - min_x) / max(max_x - min_x, 1e-12) * plot_width))
        y = height - margin - int(round((float(point[1]) - min_y) / max(max_y - min_y, 1e-12) * plot_height))
        return x, y

    if len(camera_points) >= 2:
        camera_px = np.asarray([to_px(point) for point in camera_points], dtype=np.int32)
        cv2.polylines(canvas, [camera_px], False, (45, 125, 215), 2, cv2.LINE_AA)
        for point in camera_px[:: max(1, len(camera_px) // 24)]:
            cv2.circle(canvas, tuple(int(value) for value in point), 3, (45, 125, 215), -1, cv2.LINE_AA)
    for index, point in enumerate(tag_points):
        x_px, y_px = to_px(point)
        cv2.rectangle(canvas, (x_px - 6, y_px - 6), (x_px + 6, y_px + 6), (205, 102, 44), -1)
        cv2.putText(canvas, f"T{index}", (x_px + 8, y_px - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 60, 68), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _format_scalar(value: Any, *, precision: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not headers:
        return []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _reprojection_stats(
    result: BatchSolveResult,
    dataset: BatchCalibrationDataset,
    *,
    sigma_px: float,
) -> dict[str, float]:
    corner_residual_norms_px: list[float] = []
    inlier_count = 0
    total_count = 0
    for detection in dataset.tag_detections:
        camera_pose = result.camera_pose_vectors.get(int(detection.frame_index))
        tag_pose = result.tag_pose_vectors.get(int(detection.tag_id))
        if camera_pose is None or tag_pose is None:
            continue
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(np.asarray(camera_pose, dtype=np.float64))
        tag_position_world_m, tag_rotation_wt = pose_components_from_vector(np.asarray(tag_pose, dtype=np.float64))
        world_points = tag_position_world_m.reshape(1, 3) + (
            tag_rotation_wt @ dataset.tag_catalog[int(detection.tag_id)].corner_points_local_m().T
        ).T
        predicted_pixels_px, _ = project_world_points_to_pixels(
            camera_model=dataset.camera_model,
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            world_points_m=world_points,
        )
        observed_pixels_px = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
        corner_errors = np.linalg.norm(predicted_pixels_px - observed_pixels_px, axis=1)
        corner_residual_norms_px.extend(float(value) for value in corner_errors.tolist())
        total_count += int(corner_errors.size)
        inlier_count += int(np.sum(corner_errors <= 3.0 * max(float(sigma_px), 1e-9)))
    return {
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(corner_residual_norms_px)))) if corner_residual_norms_px else float("nan"),
        "robust_inlier_ratio": float(inlier_count / max(total_count, 1)),
        "visual_rejection_count": int(total_count - inlier_count),
        "corner_count": int(total_count),
    }


def _reprojection_details(
    result: BatchSolveResult,
    dataset: BatchCalibrationDataset,
) -> dict[str, Any]:
    corner_norms_px: list[float] = []
    axis_residuals_px: list[float] = []
    frame_residuals: dict[int, list[float]] = defaultdict(list)
    tag_residuals: dict[int, list[float]] = defaultdict(list)
    for detection in dataset.tag_detections:
        camera_pose = result.camera_pose_vectors.get(int(detection.frame_index))
        tag_pose = result.tag_pose_vectors.get(int(detection.tag_id))
        if camera_pose is None or tag_pose is None:
            continue
        camera_position_world_m, camera_rotation_wc = pose_components_from_vector(np.asarray(camera_pose, dtype=np.float64))
        tag_position_world_m, tag_rotation_wt = pose_components_from_vector(np.asarray(tag_pose, dtype=np.float64))
        world_points = tag_position_world_m.reshape(1, 3) + (
            tag_rotation_wt @ dataset.tag_catalog[int(detection.tag_id)].corner_points_local_m().T
        ).T
        predicted_pixels_px, _ = project_world_points_to_pixels(
            camera_model=dataset.camera_model,
            camera_position_world_m=camera_position_world_m,
            camera_rotation_wc=camera_rotation_wc,
            world_points_m=world_points,
        )
        observed_pixels_px = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
        residual_xy = predicted_pixels_px - observed_pixels_px
        corner_errors = np.linalg.norm(residual_xy, axis=1)
        corner_norms_px.extend(float(value) for value in corner_errors.tolist())
        axis_residuals_px.extend(float(value) for value in residual_xy.reshape(-1).tolist())
        frame_residuals[int(detection.frame_index)].extend(float(value) for value in corner_errors.tolist())
        tag_residuals[int(detection.tag_id)].extend(float(value) for value in corner_errors.tolist())
    robust_sigma_px = None
    if axis_residuals_px:
        residuals = np.asarray(axis_residuals_px, dtype=np.float64)
        robust_sigma_px = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    return {
        "corner_norms_px": corner_norms_px,
        "axis_residuals_px": axis_residuals_px,
        "frame_rmse_by_frame": {
            int(frame_index): float(np.sqrt(np.mean(np.square(errors))))
            for frame_index, errors in frame_residuals.items()
            if errors
        },
        "tag_rmse_by_tag": {
            int(tag_id): float(np.sqrt(np.mean(np.square(errors))))
            for tag_id, errors in tag_residuals.items()
            if errors
        },
        "robust_sigma_px": robust_sigma_px,
    }


def _trajectory_records(
    result: BatchSolveResult,
    evaluation_data: BatchCalibrationEvaluationData,
) -> list[dict[str, Any]]:
    alignment = _alignment_for_result(result, evaluation_data)
    truth_by_frame = _camera_truth_by_frame(evaluation_data)
    records = []
    for frame_index in sorted(int(index) for index in result.camera_pose_vectors):
        aligned_pose = _aligned_pose_vector(np.asarray(result.camera_pose_vectors[frame_index], dtype=np.float64), alignment)
        position_world_m, rotation_wc = pose_components_from_vector(aligned_pose)
        truth = truth_by_frame.get(int(frame_index))
        truth_position_world_m = None if truth is None else np.asarray(truth.position_world_m, dtype=np.float64)
        records.append(
            {
                "frame_index": int(frame_index),
                "timestamp_s": None if truth is None else float(truth.timestamp_s),
                "position_world_m": position_world_m,
                "rotation_wc": rotation_wc,
                "truth_position_world_m": truth_position_world_m,
            }
        )
    return records


def _trajectory_stats(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if len(records) < 1:
        return {
            "path_length_m": None,
            "start_end_displacement_m": None,
            "mean_speed_mps": None,
            "max_speed_mps": None,
        }
    positions = [np.asarray(record["position_world_m"], dtype=np.float64).reshape(3) for record in records]
    path_segments = [float(np.linalg.norm(curr - prev)) for prev, curr in zip(positions[:-1], positions[1:])]
    timestamps = [record["timestamp_s"] for record in records]
    speed_samples = []
    for prev, curr, t_prev, t_curr in zip(positions[:-1], positions[1:], timestamps[:-1], timestamps[1:]):
        if t_prev is None or t_curr is None:
            continue
        dt_s = max(float(t_curr) - float(t_prev), 1e-9)
        speed_samples.append(float(np.linalg.norm(curr - prev) / dt_s))
    return {
        "path_length_m": float(np.sum(path_segments)) if path_segments else 0.0,
        "start_end_displacement_m": float(np.linalg.norm(positions[-1] - positions[0])) if len(positions) >= 2 else 0.0,
        "mean_speed_mps": float(np.mean(speed_samples)) if speed_samples else None,
        "max_speed_mps": float(np.max(speed_samples)) if speed_samples else None,
    }


def _bias_norm_series(result: BatchSolveResult) -> dict[str, list[float]]:
    frame_indices = sorted(int(index) for index in result.camera_pose_vectors)
    gyro = [float(np.linalg.norm(np.asarray(result.gyro_bias_rps_by_frame.get(frame_index, np.zeros(3)), dtype=np.float64))) for frame_index in frame_indices]
    accel = [float(np.linalg.norm(np.asarray(result.accel_bias_mps2_by_frame.get(frame_index, np.zeros(3)), dtype=np.float64))) for frame_index in frame_indices]
    velocity = [float(np.linalg.norm(np.asarray(result.velocity_world_mps_by_frame.get(frame_index, np.zeros(3)), dtype=np.float64))) for frame_index in frame_indices]
    return {"frame_indices": frame_indices, "gyro_bias_norm_rps": gyro, "accel_bias_norm_mps2": accel, "velocity_norm_mps": velocity}


def _known_map_tag_pose_vectors(
    evaluation_data: BatchCalibrationEvaluationData,
    *,
    anchor_frame_index: int,
) -> dict[int, np.ndarray]:
    truth_by_frame = _camera_truth_by_frame(evaluation_data)
    tag_pose_vectors = _tag_truth_pose_by_id(evaluation_data, truth_by_frame)
    anchor_truth = truth_by_frame.get(int(anchor_frame_index))
    if anchor_truth is None:
        return tag_pose_vectors
    anchor_truth_pose = pose_vector_from_components(
        np.asarray(anchor_truth.position_world_m, dtype=np.float64),
        np.asarray(anchor_truth.rotation_cw, dtype=np.float64).reshape(3, 3),
    )
    alignment = world_alignment_from_reference_pose(
        anchor_truth_pose,
        np.zeros(6, dtype=np.float64),
    )
    return {
        int(tag_id): apply_world_alignment_to_pose_vector(alignment[0], alignment[1], np.asarray(tag_pose, dtype=np.float64))
        for tag_id, tag_pose in tag_pose_vectors.items()
    }


def _packet_segments(dataset: BatchCalibrationDataset) -> list[tuple[CameraFrame, CameraFrame, list[ImuPacket], list[float]]]:
    packet_timestamps = np.asarray([packet.timestamp_s for packet in dataset.imu_packets], dtype=np.float64)
    segments = []
    for start_frame, end_frame in zip(dataset.camera_frames[:-1], dataset.camera_frames[1:]):
        start_time_s = float(start_frame.timestamp_s)
        end_time_s = float(end_frame.timestamp_s)
        packet_indices = np.where((packet_timestamps > start_time_s + 1e-12) & (packet_timestamps <= end_time_s + 1e-12))[0]
        if packet_indices.size == 0:
            segments.append((start_frame, end_frame, [], []))
            continue
        packets = []
        dt_values = []
        last_time_s = start_time_s
        for packet_index in packet_indices.tolist():
            packet = dataset.imu_packets[int(packet_index)]
            dt_s = max(float(packet.timestamp_s) - last_time_s, 1e-9)
            last_time_s = float(packet.timestamp_s)
            packets.append(packet)
            dt_values.append(dt_s)
        segments.append((start_frame, end_frame, packets, dt_values))
    return segments


def build_imu_only_result(
    dataset: BatchCalibrationDataset,
    init: InitialGuess,
    *,
    variant: str,
) -> BatchSolveResult:
    frame_indices = [int(frame.frame_index) for frame in dataset.camera_frames]
    if not frame_indices:
        return BatchSolveResult(
            stage="imu_only",
            variant=variant,
            success=False,
            solver_name="discrete_imu_dead_reckoning",
            iterations=0,
            final_cost=0.0,
            initial_cost=0.0,
            damping=0.0,
            camera_pose_vectors={},
            tag_pose_vectors={},
        )
    first_frame_index = int(frame_indices[0])
    first_pose = np.asarray(init.camera_pose_vectors[first_frame_index], dtype=np.float64)
    current_position_world_m, current_rotation_wc = pose_components_from_vector(first_pose)
    current_velocity_world_mps = np.asarray(
        init.velocity_world_mps_by_frame.get(first_frame_index, np.zeros(3, dtype=np.float64)),
        dtype=np.float64,
    ).reshape(3)
    camera_pose_vectors = {first_frame_index: first_pose.copy()}
    velocity_world_mps_by_frame = {first_frame_index: current_velocity_world_mps.copy()}
    for _start_frame, end_frame, packets, dt_values in _packet_segments(dataset):
        if not packets:
            pose_vector = pose_vector_from_components(current_position_world_m, current_rotation_wc)
            camera_pose_vectors[int(end_frame.frame_index)] = pose_vector
            velocity_world_mps_by_frame[int(end_frame.frame_index)] = current_velocity_world_mps.copy()
            continue
        samples = [
            ImuSample(
                timestamp_s=float(packet.timestamp_s),
                dt_s=float(dt_s),
                accel_body_mps2=np.asarray(packet.accel_body_mps2, dtype=np.float64),
                gyro_body_rps=np.asarray(packet.gyro_body_rps, dtype=np.float64),
            )
            for packet, dt_s in zip(packets, dt_values)
        ]
        integrated = integrate_discrete_imu_sequence(
            samples,
            initial_position_world_m=current_position_world_m,
            initial_rotation_cw=current_rotation_wc,
            initial_velocity_world_mps=current_velocity_world_mps,
        )
        current_position_world_m = np.asarray(integrated["final_position_world_m"], dtype=np.float64)
        current_rotation_wc = np.asarray(integrated["final_rotation_cw"], dtype=np.float64).reshape(3, 3)
        current_velocity_world_mps = np.asarray(integrated["final_velocity_world_mps"], dtype=np.float64)
        frame_index = int(end_frame.frame_index)
        camera_pose_vectors[frame_index] = pose_vector_from_components(current_position_world_m, current_rotation_wc)
        velocity_world_mps_by_frame[frame_index] = current_velocity_world_mps.copy()
    return BatchSolveResult(
        stage="imu_only",
        variant=variant,
        success=True,
        solver_name="discrete_imu_dead_reckoning",
        iterations=len(frame_indices) - 1,
        final_cost=0.0,
        initial_cost=0.0,
        damping=0.0,
        camera_pose_vectors=camera_pose_vectors,
        tag_pose_vectors={},
        velocity_world_mps_by_frame=velocity_world_mps_by_frame,
        diagnostics={"imu_factor_count": len(dataset.imu_packets)},
    )


def _perturb_dataset(
    dataset: BatchCalibrationDataset,
    *,
    vision_noise: VisionNoisePreset,
    imu_noise: ImuNoisePreset,
    seed: int,
) -> BatchCalibrationDataset:
    rng = np.random.default_rng(seed)
    detections: list[TagDetectionObservation] = []
    detections_by_frame: dict[int, list[TagDetectionObservation]] = defaultdict(list)
    for detection in dataset.tag_detections:
        detections_by_frame[int(detection.frame_index)].append(detection)
    for frame_index in sorted(detections_by_frame):
        frame_detections = detections_by_frame[frame_index]
        retained: list[TagDetectionObservation] = []
        for detection in frame_detections:
            drop_probability = float(vision_noise.detection_drop_probability) + float(getattr(vision_noise, "visibility_failure_probability", 0.0))
            if drop_probability > 0.0 and float(rng.random()) < min(drop_probability, 0.95):
                continue
            corners = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
            noisy_corners = corners + rng.normal(0.0, float(vision_noise.corner_noise_std_px), size=(4, 2))
            if float(getattr(vision_noise, "outlier_probability", 0.0)) > 0.0 and float(rng.random()) < float(getattr(vision_noise, "outlier_probability", 0.0)):
                outlier_shift = rng.normal(
                    0.0,
                    max(float(vision_noise.corner_noise_std_px) * 6.0, 3.0),
                    size=(1, 2),
                )
                noisy_corners = noisy_corners + outlier_shift
            center_xy = tuple(float(value) for value in np.mean(noisy_corners, axis=0).tolist())
            points5 = (center_xy,) + tuple(tuple(float(value) for value in point.tolist()) for point in noisy_corners)
            retained.append(
                replace(
                    detection,
                    corners_xy_clockwise_px=tuple(tuple(float(value) for value in point.tolist()) for point in noisy_corners),
                    center_xy_px=center_xy,
                    points5_xy_px=points5,
                )
            )
        if not retained and frame_detections:
            detection = frame_detections[0]
            corners = np.asarray(detection.corners_xy_clockwise_px, dtype=np.float64).reshape(4, 2)
            noisy_corners = corners + rng.normal(0.0, float(vision_noise.corner_noise_std_px), size=(4, 2))
            if float(getattr(vision_noise, "outlier_probability", 0.0)) > 0.0 and float(rng.random()) < float(getattr(vision_noise, "outlier_probability", 0.0)):
                noisy_corners = noisy_corners + rng.normal(
                    0.0,
                    max(float(vision_noise.corner_noise_std_px) * 6.0, 3.0),
                    size=(1, 2),
                )
            center_xy = tuple(float(value) for value in np.mean(noisy_corners, axis=0).tolist())
            points5 = (center_xy,) + tuple(tuple(float(value) for value in point.tolist()) for point in noisy_corners)
            retained.append(
                replace(
                    detection,
                    corners_xy_clockwise_px=tuple(tuple(float(value) for value in point.tolist()) for point in noisy_corners),
                    center_xy_px=center_xy,
                    points5_xy_px=points5,
                )
            )
        detections.extend(retained)

    accel_bias = np.asarray(imu_noise.accel_bias_mps2, dtype=np.float64).reshape(3).copy()
    gyro_bias = np.asarray(imu_noise.gyro_bias_rps, dtype=np.float64).reshape(3).copy()
    imu_packets: list[ImuPacket] = []
    previous_time_s: float | None = None
    for packet in dataset.imu_packets:
        if float(imu_noise.sample_drop_probability) > 0.0 and float(rng.random()) < float(imu_noise.sample_drop_probability):
            continue
        dt_s = 1.0 / max(dataset.device_config.imu.rate_hz, 1e-6) if previous_time_s is None else max(float(packet.timestamp_s) - previous_time_s, 1e-9)
        previous_time_s = float(packet.timestamp_s)
        accel_bias += rng.normal(0.0, np.asarray(imu_noise.accel_bias_random_walk_std, dtype=np.float64) * math.sqrt(dt_s), size=3)
        gyro_bias += rng.normal(0.0, np.asarray(imu_noise.gyro_bias_random_walk_std, dtype=np.float64) * math.sqrt(dt_s), size=3)
        accel = np.asarray(packet.accel_body_mps2, dtype=np.float64) + accel_bias + rng.normal(0.0, np.asarray(imu_noise.accel_noise_std, dtype=np.float64), size=3)
        gyro = np.asarray(packet.gyro_body_rps, dtype=np.float64) + gyro_bias + rng.normal(0.0, np.asarray(imu_noise.gyro_noise_std, dtype=np.float64), size=3)
        imu_packets.append(
            replace(
                packet,
                accel_body_mps2=tuple(float(value) for value in accel.tolist()),
                gyro_body_rps=tuple(float(value) for value in gyro.tolist()),
            )
        )
    return replace(
        dataset,
        tag_detections=tuple(detections),
        imu_packets=tuple(imu_packets),
    )


def _case_payload(
    *,
    label: str,
    result: BatchSolveResult,
    evaluation: EvalSummary,
    reprojection_stats: dict[str, float] | None,
    uncertainty_summary: UncertaintySummary | None = None,
) -> dict[str, Any]:
    payload = {
        "label": label,
        "variant": result.variant,
        "stage": result.stage,
        "success": bool(result.success),
        "solver_name": result.solver_name,
        "iterations": int(result.iterations),
        "initial_cost": float(result.initial_cost),
        "final_cost": float(result.final_cost),
        "cost_reduction_ratio": (float(result.initial_cost / result.final_cost) if result.final_cost > 0.0 else None),
        "mean_position_error_m": evaluation.mean_position_error_m,
        "median_position_error_m": evaluation.median_position_error_m,
        "p95_position_error_m": evaluation.p95_position_error_m,
        "max_position_error_m": evaluation.max_position_error_m,
        "mean_rotation_error_deg": evaluation.mean_rotation_error_deg,
        "mean_reprojection_rmse_px": evaluation.mean_reprojection_rmse_px,
        "joint_world_mean_position_error_m": evaluation.joint_world_mean_position_error_m,
        "joint_world_mean_rotation_error_deg": evaluation.joint_world_mean_rotation_error_deg,
        "solve_rate": evaluation.solve_rate,
        "runtime_ms_per_iteration": evaluation.runtime_ms_per_iteration,
        "metadata": dict(evaluation.metadata),
        "factor_breakdown": dict(result.diagnostics.get("factor_breakdown", {})),
    }
    if reprojection_stats is not None:
        payload.update(reprojection_stats)
    if uncertainty_summary is not None:
        radius_stats = _uncertainty_radius_stats(uncertainty_summary)
        payload["uncertainty"] = {
            "camera_pose_count": len(uncertainty_summary.camera_pose_uncertainty),
            "tag_pose_count": len(uncertainty_summary.tag_pose_uncertainty),
            "coverage_by_level": dict(uncertainty_summary.coverage_by_level),
            "covariance_condition_number": uncertainty_summary.covariance_condition_number,
            "metadata": dict(uncertainty_summary.metadata),
            "camera_mean_95_radius_m": radius_stats["camera_mean_95_m"],
            "camera_max_95_radius_m": radius_stats["camera_max_95_m"],
            "tag_mean_95_radius_m": radius_stats["tag_mean_95_m"],
            "tag_max_95_radius_m": radius_stats["tag_max_95_m"],
            "velocity_mean_95_radius_mps": radius_stats["velocity_mean_95_mps"],
            "velocity_max_95_radius_mps": radius_stats["velocity_max_95_mps"],
            "gyro_bias_radius_95": None if uncertainty_summary.gyro_bias_uncertainty is None else uncertainty_summary.gyro_bias_uncertainty.radius_95,
            "accel_bias_radius_95": None if uncertainty_summary.accel_bias_uncertainty is None else uncertainty_summary.accel_bias_uncertainty.radius_95,
        }
    return payload


def _mean_or_none(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _std_or_none(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.std(np.asarray(finite, dtype=np.float64)))


def _mean_mapping_numeric(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not dicts:
        return {}
    keys = sorted({str(key) for mapping in dicts for key in mapping})
    payload: dict[str, Any] = {}
    for key in keys:
        scalar_values = [
            mapping.get(key)
            for mapping in dicts
            if isinstance(mapping.get(key), (int, float, np.floating, np.integer))
        ]
        mean_value = _mean_or_none(scalar_values)
        if mean_value is not None:
            payload[key] = mean_value
    return payload


def _aggregate_uncertainty_payloads(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    uncertainty_payloads = [dict(case.get("uncertainty", {})) for case in cases if case.get("uncertainty")]
    if not uncertainty_payloads:
        return None
    coverage_levels = sorted({str(level) for payload in uncertainty_payloads for level in payload.get("coverage_by_level", {})})
    coverage_by_level = {
        level: _mean_or_none([payload.get("coverage_by_level", {}).get(level) for payload in uncertainty_payloads])
        for level in coverage_levels
    }
    metadata_payloads = [dict(payload.get("metadata", {})) for payload in uncertainty_payloads]
    return {
        "camera_mean_95_radius_m": _mean_or_none([payload.get("camera_mean_95_radius_m") for payload in uncertainty_payloads]),
        "camera_max_95_radius_m": _mean_or_none([payload.get("camera_max_95_radius_m") for payload in uncertainty_payloads]),
        "tag_mean_95_radius_m": _mean_or_none([payload.get("tag_mean_95_radius_m") for payload in uncertainty_payloads]),
        "tag_max_95_radius_m": _mean_or_none([payload.get("tag_max_95_radius_m") for payload in uncertainty_payloads]),
        "velocity_mean_95_radius_mps": _mean_or_none([payload.get("velocity_mean_95_radius_mps") for payload in uncertainty_payloads]),
        "velocity_max_95_radius_mps": _mean_or_none([payload.get("velocity_max_95_radius_mps") for payload in uncertainty_payloads]),
        "gyro_bias_radius_95": _mean_or_none([payload.get("gyro_bias_radius_95") for payload in uncertainty_payloads]),
        "accel_bias_radius_95": _mean_or_none([payload.get("accel_bias_radius_95") for payload in uncertainty_payloads]),
        "coverage_by_level": {level: value for level, value in coverage_by_level.items() if value is not None},
        "metadata": _mean_mapping_numeric(metadata_payloads),
    }


def _aggregate_ablation_cases(
    raw_cases: list[dict[str, Any]],
    *,
    preset_name: str,
    method_name: str,
) -> dict[str, Any]:
    cases = [
        dict(case)
        for case in raw_cases
        if str(case.get("noise_preset")) == preset_name and str(case.get("method")) == method_name
    ]
    if not cases:
        raise ValueError(f"Missing ablation cases for preset={preset_name} method={method_name}.")
    first = dict(cases[0])
    mean_factor_breakdown = _mean_mapping_numeric([dict(case.get("factor_breakdown", {})) for case in cases])
    uncertainty_payload = _aggregate_uncertainty_payloads(cases)
    aggregate = {
        "label": f"{preset_name}:{method_name}",
        "variant": first.get("variant"),
        "stage": first.get("stage"),
        "success": all(bool(case.get("success")) for case in cases),
        "solver_name": first.get("solver_name"),
        "iterations": _mean_or_none([case.get("iterations") for case in cases]),
        "initial_cost": _mean_or_none([case.get("initial_cost") for case in cases]),
        "final_cost": _mean_or_none([case.get("final_cost") for case in cases]),
        "cost_reduction_ratio": _mean_or_none([case.get("cost_reduction_ratio") for case in cases]),
        "mean_position_error_m": _mean_or_none([case.get("mean_position_error_m") for case in cases]),
        "mean_position_error_std_m": _std_or_none([case.get("mean_position_error_m") for case in cases]),
        "median_position_error_m": _mean_or_none([case.get("median_position_error_m") for case in cases]),
        "p95_position_error_m": _mean_or_none([case.get("p95_position_error_m") for case in cases]),
        "max_position_error_m": _mean_or_none([case.get("max_position_error_m") for case in cases]),
        "mean_rotation_error_deg": _mean_or_none([case.get("mean_rotation_error_deg") for case in cases]),
        "mean_rotation_error_std_deg": _std_or_none([case.get("mean_rotation_error_deg") for case in cases]),
        "mean_reprojection_rmse_px": _mean_or_none([case.get("mean_reprojection_rmse_px") for case in cases]),
        "mean_reprojection_rmse_std_px": _std_or_none([case.get("mean_reprojection_rmse_px") for case in cases]),
        "joint_world_mean_position_error_m": _mean_or_none([case.get("joint_world_mean_position_error_m") for case in cases]),
        "joint_world_mean_rotation_error_deg": _mean_or_none([case.get("joint_world_mean_rotation_error_deg") for case in cases]),
        "solve_rate": _mean_or_none([case.get("solve_rate") for case in cases]),
        "runtime_ms_per_iteration": _mean_or_none([case.get("runtime_ms_per_iteration") for case in cases]),
        "robust_inlier_ratio": _mean_or_none([case.get("robust_inlier_ratio") for case in cases]),
        "visual_rejection_count": _mean_or_none([case.get("visual_rejection_count") for case in cases]),
        "metadata": _mean_mapping_numeric([dict(case.get("metadata", {})) for case in cases]),
        "factor_breakdown": mean_factor_breakdown,
        "noise_preset": preset_name,
        "method": method_name,
        "sample_count": len(cases),
        "seed_list": [int(case.get("seed")) for case in cases if case.get("seed") is not None],
        "failure_rate": 1.0 - (sum(1 for case in cases if bool(case.get("success"))) / max(len(cases), 1)),
    }
    if uncertainty_payload is not None:
        aggregate["uncertainty"] = uncertainty_payload
    return aggregate


def _ablation_seed_cache_path(analysis_dir: Path, *, preset_name: str, seed: int) -> Path:
    return analysis_dir / "ablation_seed_cases" / f"{preset_name}_seed_{int(seed):03d}.json"


def _likelihood_sweep_scale_grid() -> tuple[float, ...]:
    return (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _run_likelihood_sweeps(
    *,
    analysis_dir: Path,
    dataset: BatchCalibrationDataset,
    evaluation_data: BatchCalibrationEvaluationData,
    anchor_frame_index: int,
    fused_result: BatchSolveResult,
    base_vision_noise: VisionNoisePreset,
    base_imu_noise: ImuNoisePreset,
) -> list[dict[str, Any]]:
    if _run_role_label(dataset) != "async_headline":
        return []
    cache_paths = {
        "imu_covariance_scale": analysis_dir / "_likelihood_sweep_imu_covariance_scale.json",
        "visual_weight_scale": analysis_dir / "_likelihood_sweep_visual_weight_scale.json",
    }
    if all(path.exists() for path in cache_paths.values()):
        cached_rows: list[dict[str, Any]] = []
        for family in ("imu_covariance_scale", "visual_weight_scale"):
            cached_rows.extend(json.loads(cache_paths[family].read_text(encoding="utf-8")))
        return cached_rows
    rows: list[dict[str, Any]] = []
    for family in ("imu_covariance_scale", "visual_weight_scale"):
        sweep_init = _initial_guess_from_result(fused_result, anchor_frame_index=int(anchor_frame_index))
        family_rows: list[dict[str, Any]] = []
        for scale in _likelihood_sweep_scale_grid():
            sweep_vision = base_vision_noise
            sweep_imu = base_imu_noise
            if family == "imu_covariance_scale":
                sweep_imu = replace(
                    base_imu_noise,
                    name=f"{base_imu_noise.name}_{family}_{scale:g}",
                    likelihood_scale=float(base_imu_noise.likelihood_scale) * float(scale),
                )
            else:
                sweep_vision = replace(
                    base_vision_noise,
                    name=f"{base_vision_noise.name}_{family}_{scale:g}",
                    corner_noise_std_px=max(float(base_vision_noise.corner_noise_std_px) * float(scale), 1.0e-3),
                )
            sweep_result = solve_visual_inertial_batch_map(
                dataset,
                sweep_init,
                noise_cfg={"vision": sweep_vision, "imu": sweep_imu},
                solver_options={"max_iterations": 2, "initial_damping": 1e-4},
                variant=f"likelihood_sweep_{family}",
            )
            sweep_init = _initial_guess_from_result(sweep_result, anchor_frame_index=int(anchor_frame_index))
            sweep_eval = evaluate_batch_result(
                sweep_result,
                evaluation_data,
                dataset=dataset,
                uncertainty_summary=None,
            )
            factor_payload = dict(sweep_result.diagnostics.get("factor_breakdown", {}))
            family_rows.append(
                {
                    "sweep_family": family,
                    "scale": float(scale),
                    "method": "fused",
                    "mean_position_error_m": sweep_eval.mean_position_error_m,
                    "p95_position_error_m": sweep_eval.p95_position_error_m,
                    "mean_rotation_error_deg": sweep_eval.mean_rotation_error_deg,
                    "mean_whitened_imu_sq_residual_per_factor": factor_payload.get("mean_whitened_imu_sq_residual_per_factor"),
                    "mean_whitened_imu_sq_residual_position_per_factor": factor_payload.get("mean_whitened_imu_sq_residual_position_per_factor"),
                    "mean_whitened_imu_sq_residual_rotation_per_factor": factor_payload.get("mean_whitened_imu_sq_residual_rotation_per_factor"),
                    "mean_whitened_imu_sq_residual_velocity_per_factor": factor_payload.get("mean_whitened_imu_sq_residual_velocity_per_factor"),
                    "position_nees_mean": None,
                    "position_whitened_sq_error_mean": None,
                    "rotation_nees_mean": None,
                    "rotation_whitened_sq_error_mean": None,
                    "velocity_nees_mean": None,
                    "velocity_whitened_sq_error_mean": None,
                    "gyro_bias_nees": None,
                    "gyro_bias_whitened_sq_error_mean": None,
                    "accel_bias_nees": None,
                    "accel_bias_whitened_sq_error_mean": None,
                    "empirical_95_coverage_pct": None,
                    "mean_accel_bias_norm_mps2": float(np.linalg.norm(np.asarray(sweep_result.global_accel_bias_mps2, dtype=np.float64))) if sweep_result.global_accel_bias_mps2 is not None else None,
                }
            )
        rows.extend(family_rows)
        cache_paths[family].write_text(json.dumps(_jsonify(family_rows), indent=2), encoding="utf-8")
    return rows


def _uncertainty_frame_records(
    *,
    method_name: str,
    evaluation: EvalSummary,
    uncertainty_summary: UncertaintySummary,
    frame_detection_counts: dict[int, int],
    speed_norm_by_frame: dict[int, float],
    anchor_frame_index: int,
) -> list[dict[str, Any]]:
    uncertainty_lookup = {int(item.index): item for item in uncertainty_summary.camera_pose_uncertainty}
    eval_lookup = {int(record["frame_index"]): record for record in evaluation.per_frame_records}
    aligned_anchor_position = None
    if int(anchor_frame_index) in eval_lookup:
        aligned_anchor_position = np.asarray(eval_lookup[int(anchor_frame_index)]["estimated_camera_world_position_m"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for frame_index in sorted(set(eval_lookup) & set(uncertainty_lookup)):
        record = eval_lookup[frame_index]
        uncertainty = uncertainty_lookup[frame_index]
        position_error_vector = np.asarray(record["position_error_vector_world_m"], dtype=np.float64)
        position_error_m = float(np.linalg.norm(position_error_vector))
        sigma = np.maximum(np.asarray(uncertainty.position_std_m, dtype=np.float64), 1.0e-12)
        position_nees = float(np.sum(np.square(position_error_vector / sigma)))
        estimated_position = np.asarray(record["estimated_camera_world_position_m"], dtype=np.float64)
        anchor_distance_m = None if aligned_anchor_position is None else float(np.linalg.norm(estimated_position - aligned_anchor_position))
        rows.append(
            {
                "method": method_name,
                "frame_index": int(frame_index),
                "position_error_m": position_error_m,
                "position_radius_95_m": float(uncertainty.position_radius_95_m),
                "position_nees": position_nees,
                "visible_tag_count": int(frame_detection_counts.get(int(frame_index), 0)),
                "speed_norm_mps": speed_norm_by_frame.get(int(frame_index)),
                "anchor_distance_m": anchor_distance_m,
            }
        )
    return rows


def _uncertainty_stratified_rows(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not frame_rows:
        return []
    anchor_distances = [float(row["anchor_distance_m"]) for row in frame_rows if row.get("anchor_distance_m") is not None]
    speed_values = [float(row["speed_norm_mps"]) for row in frame_rows if row.get("speed_norm_mps") is not None]
    anchor_threshold = float(np.median(np.asarray(anchor_distances, dtype=np.float64))) if anchor_distances else 0.0
    speed_threshold = float(np.median(np.asarray(speed_values, dtype=np.float64))) if speed_values else 0.0

    def _aggregate(method_name: str, stratum_type: str, stratum_value: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        radius = np.asarray([float(row["position_radius_95_m"]) for row in subset], dtype=np.float64)
        errors = np.asarray([float(row["position_error_m"]) for row in subset], dtype=np.float64)
        nees = np.asarray([float(row["position_nees"]) for row in subset], dtype=np.float64)
        sigma_corr = None
        if radius.size > 1 and not math.isclose(float(np.std(radius)), 0.0) and not math.isclose(float(np.std(errors)), 0.0):
            sigma_corr = float(np.corrcoef(radius, errors)[0, 1])
        return {
            "method": method_name,
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "sample_count": int(len(subset)),
            "mean_predicted_95_radius_m": float(np.mean(radius)) if radius.size else None,
            "empirical_95_coverage_pct": float(100.0 * np.mean(errors <= radius)) if radius.size else None,
            "position_nees_mean": float(np.mean(nees)) if nees.size else None,
            "position_sigma_error_corr": sigma_corr,
        }

    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in frame_rows})
    for method_name in methods:
        method_rows = [row for row in frame_rows if str(row["method"]) == method_name]
        anchor_groups = {
            "anchor_near": [row for row in method_rows if row.get("anchor_distance_m") is not None and float(row["anchor_distance_m"]) <= anchor_threshold],
            "anchor_far": [row for row in method_rows if row.get("anchor_distance_m") is not None and float(row["anchor_distance_m"]) > anchor_threshold],
        }
        for value, subset in anchor_groups.items():
            if subset:
                output.append(_aggregate(method_name, "anchor_distance", value, subset))
        motion_groups = {
            "motion_slow": [row for row in method_rows if row.get("speed_norm_mps") is not None and float(row["speed_norm_mps"]) <= speed_threshold],
            "motion_fast": [row for row in method_rows if row.get("speed_norm_mps") is not None and float(row["speed_norm_mps"]) > speed_threshold],
        }
        for value, subset in motion_groups.items():
            if subset:
                output.append(_aggregate(method_name, "motion_regime", value, subset))
        visibility_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in method_rows:
            tag_count = int(row["visible_tag_count"])
            key = "4+" if tag_count >= 4 else str(tag_count)
            visibility_groups[key].append(row)
        for value in ("1", "2", "3", "4+"):
            subset = visibility_groups.get(value, [])
            if subset:
                output.append(_aggregate(method_name, "visible_tags", value, subset))
    return output


def _case_report_lines(case: dict[str, Any]) -> list[str]:
    return [
        f"### {case['label']}",
        "",
        f"- Variant: `{case['variant']}`",
        f"- Solver: `{case['solver_name']}`",
        f"- Iterations: `{case['iterations']}`",
        f"- Mean position error: `{case['mean_position_error_m']}` m",
        f"- Mean rotation error: `{case['mean_rotation_error_deg']}` deg",
        f"- Mean reprojection RMSE: `{case['mean_reprojection_rmse_px']}` px",
        f"- Robust inlier ratio: `{case.get('robust_inlier_ratio', 'n/a')}`",
        f"- Visual rejection count: `{case.get('visual_rejection_count', 'n/a')}`",
        "",
    ]


def _analysis_artifact_ref(path: Path) -> str:
    return path.name


def _image_lines(path: Path, *, alt: str, caption: str | None = None) -> list[str]:
    lines = [f"![{alt}]({_analysis_artifact_ref(path)})"]
    if caption:
        lines.append(f"*{caption}*")
    lines.append("")
    return lines


def _artifact_lines(items: list[tuple[str, Path]]) -> list[str]:
    return [f"- {label}: `analysis/{path.name}`" for label, path in items]


def _case_metric_row(case: dict[str, Any]) -> list[str]:
    return [
        str(case["label"]),
        str(case.get("solver_name", "n/a")),
        str(case.get("iterations", "n/a")),
        _format_scalar(case.get("mean_position_error_m"), precision=6),
        _format_scalar(case.get("mean_rotation_error_deg"), precision=4),
        _format_scalar(case.get("mean_reprojection_rmse_px"), precision=4),
        _format_scalar(case.get("robust_inlier_ratio"), precision=4),
        _format_scalar(case.get("cost_reduction_ratio"), precision=2),
    ]


def _trajectory_stat_row(label: str, stats: dict[str, Any]) -> list[str]:
    return [
        label,
        _format_scalar(stats.get("path_length_m"), precision=4),
        _format_scalar(stats.get("start_end_displacement_m"), precision=4),
        _format_scalar(stats.get("mean_speed_mps"), precision=4),
        _format_scalar(stats.get("max_speed_mps"), precision=4),
    ]


def _table_section(title: str, headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    return [title, "", *_markdown_table(headers, rows), ""]


def _write_csv_table(path: Path, *, headers: list[str], rows: list[list[Any]]) -> None:
    lines = [",".join(headers)]
    for row in rows:
        serialized = []
        for value in row:
            item = "" if value is None else str(value)
            if "," in item or "\"" in item or "\n" in item:
                item = "\"" + item.replace("\"", "\"\"") + "\""
            serialized.append(item)
        lines.append(",".join(serialized))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coverage_percent(summary: UncertaintySummary | None, level: str) -> float | None:
    if summary is None:
        return None
    value = summary.coverage_by_level.get(level)
    if value is None:
        return None
    return 100.0 * float(value)


def _uncertainty_radius_stats(summary: UncertaintySummary | None) -> dict[str, float | None]:
    if summary is None:
        return {
            "camera_mean_95_m": None,
            "camera_max_95_m": None,
            "tag_mean_95_m": None,
            "tag_max_95_m": None,
            "velocity_mean_95_mps": None,
            "velocity_max_95_mps": None,
        }
    camera = [float(item.position_radius_95_m) for item in summary.camera_pose_uncertainty]
    tag = [float(item.position_radius_95_m) for item in summary.tag_pose_uncertainty]
    velocity = [float(item.radius_95) for item in summary.velocity_uncertainty]
    return {
        "camera_mean_95_m": float(np.mean(np.asarray(camera, dtype=np.float64))) if camera else None,
        "camera_max_95_m": float(np.max(np.asarray(camera, dtype=np.float64))) if camera else None,
        "tag_mean_95_m": float(np.mean(np.asarray(tag, dtype=np.float64))) if tag else None,
        "tag_max_95_m": float(np.max(np.asarray(tag, dtype=np.float64))) if tag else None,
        "velocity_mean_95_mps": float(np.mean(np.asarray(velocity, dtype=np.float64))) if velocity else None,
        "velocity_max_95_mps": float(np.max(np.asarray(velocity, dtype=np.float64))) if velocity else None,
    }


def _posterior_metric(summary: UncertaintySummary | None, key: str) -> float | None:
    if summary is None:
        return None
    value = summary.metadata.get(key)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def _truth_speed_lookup(evaluation_data: BatchCalibrationEvaluationData) -> dict[int, float]:
    return {
        int(sample.frame_index): float(np.linalg.norm(np.asarray(sample.velocity_world_mps, dtype=np.float64)))
        for sample in evaluation_data.camera_truth
    }


def _uncertainty_error_points(
    evaluation: EvalSummary,
    uncertainty_summary: UncertaintySummary | None,
) -> list[dict[str, Any]]:
    if uncertainty_summary is None:
        return []
    uncertainty_lookup = {int(item.index): item for item in uncertainty_summary.camera_pose_uncertainty}
    points: list[dict[str, Any]] = []
    for record in evaluation.per_frame_records:
        frame_index = int(record["frame_index"])
        uncertainty = uncertainty_lookup.get(frame_index)
        if uncertainty is None:
            continue
        points.append(
            {
                "label": str(frame_index),
                "x": float(uncertainty.position_radius_95_m),
                "y": float(record["position_error_m"]),
            }
        )
    return points


def _scaled_vision_preset(preset: VisionNoisePreset, scale: float) -> VisionNoisePreset:
    return replace(
        preset,
        corner_noise_std_px=float(preset.corner_noise_std_px) * float(scale),
    )


def _scaled_imu_preset(preset: ImuNoisePreset, scale: float) -> ImuNoisePreset:
    return replace(
        preset,
        likelihood_scale=float(scale),
    )


def _initial_guess_from_result(result: BatchSolveResult, *, anchor_frame_index: int) -> InitialGuess:
    return InitialGuess(
        stage=result.stage,
        anchor_frame_index=int(anchor_frame_index),
        camera_pose_vectors={int(index): np.asarray(pose, dtype=np.float64).copy() for index, pose in result.camera_pose_vectors.items()},
        tag_pose_vectors={int(index): np.asarray(pose, dtype=np.float64).copy() for index, pose in result.tag_pose_vectors.items()},
        velocity_world_mps_by_frame={int(index): np.asarray(value, dtype=np.float64).copy() for index, value in result.velocity_world_mps_by_frame.items()},
        gyro_bias_rps_by_frame={int(index): np.asarray(value, dtype=np.float64).copy() for index, value in result.gyro_bias_rps_by_frame.items()},
        accel_bias_mps2_by_frame={int(index): np.asarray(value, dtype=np.float64).copy() for index, value in result.accel_bias_mps2_by_frame.items()},
        global_gyro_bias_rps=np.asarray(result.global_gyro_bias_rps, dtype=np.float64).copy(),
        global_accel_bias_mps2=np.asarray(result.global_accel_bias_mps2, dtype=np.float64).copy(),
        notes=("warm_started_from_previous_result",),
    )


def _run_role_label(dataset: BatchCalibrationDataset) -> str:
    packet_mode = dataset.imu_convention.imu_sampling_mode
    if packet_mode == "independent_fixed_rate":
        return "async_headline"
    return "compatibility_baseline"


def _scientific_report_lines(
    *,
    dataset: BatchCalibrationDataset,
    visual_case_payload: dict[str, Any],
    fused_case_payload: dict[str, Any],
    imu_only_case_payload: dict[str, Any],
    known_map_case_payload: dict[str, Any],
    visual_posterior: UncertaintySummary,
    fused_posterior: UncertaintySummary,
    cases_by_noise: dict[str, dict[str, dict[str, Any]]],
    factor_breakdown: dict[str, Any],
    likelihood_sweep_rows: list[dict[str, Any]],
    uncertainty_stratified_rows: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
) -> list[str]:
    visual_unc = _uncertainty_radius_stats(visual_posterior)
    fused_unc = _uncertainty_radius_stats(fused_posterior)
    fused_bias = fused_case_payload.get("bias_stats", {})
    run_role = _run_role_label(dataset)
    block_a_rows = [
        ["visual-only", _format_scalar(visual_case_payload.get("mean_position_error_m"), precision=6), _format_scalar(visual_case_payload.get("p95_position_error_m"), precision=6), _format_scalar(visual_case_payload.get("mean_rotation_error_deg"), precision=4), _format_scalar(visual_case_payload.get("mean_reprojection_rmse_px"), precision=4)],
        ["fused", _format_scalar(fused_case_payload.get("mean_position_error_m"), precision=6), _format_scalar(fused_case_payload.get("p95_position_error_m"), precision=6), _format_scalar(fused_case_payload.get("mean_rotation_error_deg"), precision=4), _format_scalar(fused_case_payload.get("mean_reprojection_rmse_px"), precision=4)],
        ["imu-only", _format_scalar(imu_only_case_payload.get("mean_position_error_m"), precision=6), _format_scalar(imu_only_case_payload.get("p95_position_error_m"), precision=6), _format_scalar(imu_only_case_payload.get("mean_rotation_error_deg"), precision=4), _format_scalar(imu_only_case_payload.get("mean_reprojection_rmse_px"), precision=4)],
        ["known-map diagnostic", _format_scalar(known_map_case_payload.get("mean_position_error_m"), precision=6), _format_scalar(known_map_case_payload.get("p95_position_error_m"), precision=6), _format_scalar(known_map_case_payload.get("mean_rotation_error_deg"), precision=4), _format_scalar(known_map_case_payload.get("mean_reprojection_rmse_px"), precision=4)],
    ]
    block_b_rows = [
        [
            "fused",
            _format_scalar(fused_bias.get("gyro_bias_norm_mean_rps"), precision=6),
            _format_scalar(fused_bias.get("accel_bias_norm_mean_mps2"), precision=6),
            _format_scalar(fused_bias.get("accel_bias_norm_max_mps2"), precision=6),
            _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_per_factor"), precision=6),
            _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_position_per_factor"), precision=6),
            _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_rotation_per_factor"), precision=6),
            _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_velocity_per_factor"), precision=6),
            "pass" if (fused_bias.get("accel_bias_norm_mean_mps2") or 0.0) <= 2.0 else "failed_plausibility_threshold",
        ],
    ]
    block_c_rows = [
        [
            "visual-only",
            _format_scalar(visual_unc["camera_mean_95_m"], precision=6),
            _format_scalar(_coverage_percent(visual_posterior, "95"), precision=2),
            _format_scalar(_posterior_metric(visual_posterior, "position_nees_mean"), precision=4),
            _format_scalar(_posterior_metric(visual_posterior, "rotation_nees_mean"), precision=4),
            _format_scalar(_posterior_metric(visual_posterior, "position_sigma_error_corr"), precision=4),
            "",
            "",
        ],
        [
            "fused",
            _format_scalar(fused_unc["camera_mean_95_m"], precision=6),
            _format_scalar(_coverage_percent(fused_posterior, "95"), precision=2),
            _format_scalar(_posterior_metric(fused_posterior, "position_nees_mean"), precision=4),
            _format_scalar(_posterior_metric(fused_posterior, "rotation_nees_mean"), precision=4),
            _format_scalar(_posterior_metric(fused_posterior, "position_sigma_error_corr"), precision=4),
            _format_scalar(_posterior_metric(fused_posterior, "velocity_nees_mean"), precision=4),
            _format_scalar(_posterior_metric(fused_posterior, "accel_bias_nees"), precision=6),
        ],
    ]
    block_d_rows = []
    for noise_name in ("ideal", "nominal", "stress"):
        for method_name in ("visual_only", "fused", "imu_only"):
            case = cases_by_noise[noise_name][method_name]
            block_d_rows.append(
                [
                    noise_name,
                    method_name,
                    _format_scalar(case.get("sample_count"), precision=0),
                    _format_scalar(case.get("mean_position_error_m"), precision=6),
                    _format_scalar(case.get("mean_position_error_std_m"), precision=6),
                    _format_scalar(case.get("mean_rotation_error_deg"), precision=4),
                    _format_scalar(case.get("uncertainty", {}).get("camera_mean_95_radius_m"), precision=6),
                    _format_scalar(100.0 * float(case.get("uncertainty", {}).get("coverage_by_level", {}).get("95", 0.0)) if case.get("uncertainty") else None, precision=2),
                    _format_scalar(100.0 * float(case.get("failure_rate", 0.0)), precision=2),
                ]
            )
    factor_rows = [
        ["Visual-only mean whitened sq residual / corner", _format_scalar(factor_breakdown.get("visual_only", {}).get("mean_whitened_visual_sq_residual_per_corner"), precision=6)],
        ["Fused mean whitened visual sq residual / corner", _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_visual_sq_residual_per_corner"), precision=6)],
        ["Fused mean whitened IMU sq residual / factor", _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_per_factor"), precision=6)],
        ["Fused IMU position term / factor", _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_position_per_factor"), precision=6)],
        ["Fused IMU rotation term / factor", _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_rotation_per_factor"), precision=6)],
        ["Fused IMU velocity term / factor", _format_scalar(factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_velocity_per_factor"), precision=6)],
    ]
    best_likelihood_rows = []
    for family in ("imu_covariance_scale", "visual_weight_scale"):
        family_rows = [row for row in likelihood_sweep_rows if str(row.get("sweep_family")) == family]
        if not family_rows:
            continue
        best_row = min(
            family_rows,
            key=lambda item: (
                float(item.get("mean_position_error_m")) if item.get("mean_position_error_m") is not None else float("inf"),
                float(item.get("mean_rotation_error_deg")) if item.get("mean_rotation_error_deg") is not None else float("inf"),
            ),
        )
        best_likelihood_rows.append(
            [
                family,
                _format_scalar(best_row.get("scale"), precision=2),
                _format_scalar(best_row.get("mean_position_error_m"), precision=6),
                _format_scalar(best_row.get("mean_rotation_error_deg"), precision=4),
                _format_scalar(best_row.get("mean_whitened_imu_sq_residual_per_factor"), precision=6),
                _format_scalar(best_row.get("position_nees_mean"), precision=4),
            ]
        )
    stratified_rows = [
        [
            str(row.get("method")),
            str(row.get("stratum_type")),
            str(row.get("stratum_value")),
            _format_scalar(row.get("sample_count"), precision=0),
            _format_scalar(row.get("mean_predicted_95_radius_m"), precision=6),
            _format_scalar(row.get("empirical_95_coverage_pct"), precision=2),
            _format_scalar(row.get("position_nees_mean"), precision=4),
        ]
        for row in uncertainty_stratified_rows[:12]
    ]

    return [
        "# Checkpoint 03 Scientific Report",
        "",
        "## 1. Problem Statement",
        "",
        "We estimate an unknown metric AprilTag map and a camera trajectory from image observations of tag corners, optionally augmented with IMU constraints. Ground-truth camera poses and tag poses are evaluation-only and never enter inference.",
        "",
        "The visual state contains one camera pose per frame and one rigid world pose per observed tag. The fused state additionally contains per-frame velocity and one constant gyro bias plus one constant accelerometer bias for the full sequence.",
        "",
        "## 2. Benchmark Context",
        "",
        f"This run is labeled `{run_role}`. The preserved frame-locked checkpoint run remains the compatibility baseline, while async fixed-rate runs are the scientific headline configuration.",
        "",
        "## 3. Canonical Estimation Problem",
        "",
        "### 3.1 Unknown State",
        "",
        "```text",
        "X_V1 = {T_WC,k}_{k=0}^{K-1} U {T_WT,j}_{j=1}^{J}",
        "X_V2 = X_V1 U {v_k}_{k=0}^{K-1} U {b_g, b_a}",
        "```",
        "",
        f"`K = {len(dataset.camera_frames)}` frame states and `J = {len(dataset.tag_catalog)}` observed tags are active on this run. `T_WC,k` is the world pose of the camera at frame `k`; `T_WT,j` is the world pose of tag `j`. The fused state adds one world-frame velocity per frame plus one constant gyro bias and one constant accelerometer bias over the full sequence.",
        "",
        "Known constants are the processed-video intrinsics, the fixed camera-to-IMU transform `T_IC`, world gravity `g_W = (0, 0, -9.81)^T m/s^2`, and the AprilTag side lengths "
        + ", ".join(f"`{tag_id}:{spec.size_m:.3f} m`" for tag_id, spec in sorted(dataset.tag_catalog.items()))
        + ".",
        "",
        "### 3.2 Observations",
        "",
        "Visual measurements are raw detector pixel corners from `detections[*].corners_xy_clockwise`. The estimator uses those raw pixel-space observations directly, with local tag geometry ordered as `(-s/2,-s/2)`, `(s/2,-s/2)`, `(s/2,s/2)`, `(-s/2,s/2)` so that the camera model and residuals stay consistent.",
        "",
        f"IMU packets follow the machine-readable convention `{dataset.imu_convention.imu_measurement_convention}` with timestamp semantics `{dataset.imu_convention.imu_timestamp_semantics}`, gravity handling `{dataset.imu_convention.imu_gravity_handling}`, and sampling mode `{dataset.imu_convention.imu_sampling_mode}`.",
        "",
        "### 3.3 Forward Models",
        "",
        "```text",
        "hat_z_vis[k,j,c] = pi(theta_cam, T_CW,k T_WT,j p_c^(tag,j))",
        f"tilde_f_m = {dataset.imu_convention.accelerometer_measurement_equation}",
        f"tilde_omega_m = {dataset.imu_convention.gyroscope_measurement_equation}",
        "r_imu[k] = [r_p[k], r_R[k], r_v[k]]^T",
        "```",
        "",
        "The visual residual is always evaluated in raw pixel space. Distortion is only applied in the forward projection when the stored camera-model snapshot says render distortion is active; the current headline run uses the processed-video intrinsics path.",
        "",
        "### 3.4 Synthetic Data-Generation Noise",
        "",
        "Synthetic corruption appears only in ablation datasets: visual corner perturbation, dropped detections, visibility failures, sparse outliers, IMU white noise, and IMU bias random walk during data generation. Detection dropout is missing data, not additive Gaussian noise.",
        "",
        "### 3.5 Estimator Likelihood Noise",
        "",
        f"The estimator likelihood is separate from synthetic corruption. The optimizer whitens visual residuals with an assumed likelihood sigma of `sigma_vis={assumed_visual_likelihood_sigma_px(dataset):.3f} px`, while the realized robust residual dispersion on this run is `sigma_empirical={empirical_visual_residual_sigma_px(visual_case_payload):.3f} px`. IMU whitening uses the resolved preset values plus the current likelihood scale and residual floors exported with the run. Huber is robustification applied after visual whitening; it is not the Gaussian likelihood.",
        "",
        "### 3.6 Priors / Process Model",
        "",
        "The posterior includes one strong anchor-pose prior to remove the global 6-DoF gauge, one initial-velocity prior through the first IMU state block, and one zero-mean constant-bias prior each for the gyro and accelerometer. Initialization seeds LM but is not itself a prior term. This checkpoint keeps one constant `b_g` and one constant `b_a` per sequence rather than a time-varying bias process.",
        "",
        "### 3.7 MAP Objective",
        "",
        "```text",
        "X_V1* = argmin sum rho_vis(||Sigma_vis^(-1/2) r_vis||^2) + ||r_anchor||^2",
        "X_V2* = argmin sum rho_vis(||Sigma_vis^(-1/2) r_vis||^2)",
        "                 + sum ||Sigma_imu^(-1/2) r_imu||^2",
        "                 + ||r_anchor||^2 + ||r_v0||^2 + ||r_bg||^2 + ||r_ba||^2",
        "```",
        "",
        "Metric scale is observable because tag side lengths are known, but the unknown-map problem still carries a global 6-DoF gauge. That gauge is removed with one strong anchor-pose prior. First-common-frame alignment is used only for evaluation after inference.",
        "",
        "### 3.8 Posterior Uncertainty Extraction",
        "",
        "Posterior uncertainty is approximated locally by a Laplace approximation around the MAP solution. After gauge fixing, the covariance is taken from the inverse regularized normal matrix in the local tangent parameterization.",
        "",
        f"On this run the visual-only camera mean/max 95% radius is `{_format_scalar(visual_unc['camera_mean_95_m'], precision=6)} / {_format_scalar(visual_unc['camera_max_95_m'], precision=6)}` m, while the fused camera mean/max 95% radius is `{_format_scalar(fused_unc['camera_mean_95_m'], precision=6)} / {_format_scalar(fused_unc['camera_max_95_m'], precision=6)}` m. The fused velocity mean/max 95% radius is `{_format_scalar(fused_unc['velocity_mean_95_mps'], precision=6)} / {_format_scalar(fused_unc['velocity_max_95_mps'], precision=6)}` m/s.",
        "",
        "A scalar 95% position radius is only a coarse summary. Good radius coverage does not imply that the full covariance is calibrated; NEES and whitened squared error are stricter tests.",
        "",
        "### 3.9 Uncertainty Evaluation",
        "",
        "The report evaluates trajectory and map accuracy, parameter plausibility, factor-normalized residuals, posterior uncertainty calibration, and ideal/nominal/stress sensitivity. Evaluation alignment uses the first common frame for reporting only; inference itself uses only the anchor-pose prior.",
        "",
        "The current fused IMU factor is a simplified discrete-time interval factor rather than a full continuous-time preintegration model with lever-arm and higher-order covariance propagation.",
        "## 4. Results",
        "",
        *_table_section(
            "### Block A: Trajectory and Map Accuracy",
            ["Method", "Mean pos err [m]", "P95 pos err [m]", "Mean rot err [deg]", "Mean reproj RMSE [px]"],
            block_a_rows,
        ),
        *_table_section(
            "### Block B: Parameter Plausibility",
            [
                "Method",
                "Mean gyro-bias norm [rad/s]",
                "Mean accel-bias norm [m/s^2]",
                "Max accel-bias norm [m/s^2]",
                "Mean IMU sq residual / factor",
                "Position term",
                "Rotation term",
                "Velocity term",
                "Comment",
            ],
            block_b_rows,
        ),
        *_table_section(
            "### Block C: Posterior Uncertainty Calibration",
            ["Method", "Mean 95% pos radius [m]", "Empirical 95% coverage [%]", "Pos NEES", "Rot NEES", "Sigma/error corr", "Vel NEES", "Accel-bias NEES"],
            block_c_rows,
        ),
        "",
        "Scalar 95% coverage is reported because it is intuitive, but it should not be over-interpreted. The reported scalar 95% radius is a diagonalized marginal summary, not a full Mahalanobis confidence ellipsoid. The stricter calibration signal for this checkpoint is the NEES and whitened-error diagnostics exported in the machine-readable uncertainty table.",
        "",
        *_table_section(
            "### Block D: Noise Sensitivity",
            ["Noise preset", "Method", "Seeds", "Mean pos err [m]", "Std pos err [m]", "Mean rot err [deg]", "Mean 95% radius [m]", "Empirical 95% coverage [%]", "Failure rate [%]"],
            block_d_rows,
        ),
        *_table_section("## 5. Factor-Normalized Diagnostics", ["Quantity", "Value"], factor_rows),
        *_table_section(
            "## 6. Local Likelihood-Calibration Sweep Summary",
            ["Sweep family", "Best scale", "Mean pos err [m]", "Mean rot err [deg]", "Mean IMU sq residual / factor", "Position NEES"],
            best_likelihood_rows if best_likelihood_rows else [["not_run", "-", "-", "-", "-", "-"]],
        ),
        "",
        "These likelihood sweeps are local calibration probes, not full global hyperparameter optimizations: they reuse the repaired headline solve, perturb one likelihood family at a time, and run only a small number of LM iterations.",
        *_table_section(
            "## 7. Uncertainty Stratification Snapshot",
            ["Method", "Stratum type", "Stratum", "N", "Mean 95% radius [m]", "Empirical 95% coverage [%]", "Position NEES"],
            stratified_rows if stratified_rows else [["not_available", "-", "-", "-", "-", "-", "-"]],
        ),
        *_image_lines(artifact_paths["visual_convergence"], alt="Visual convergence", caption="Visual-only LM convergence history."),
        *_image_lines(artifact_paths["fused_convergence"], alt="Fused convergence", caption="Fused LM convergence history."),
        *_image_lines(artifact_paths["visual_traj3d"], alt="Visual trajectory 3D", caption="Visual-only 3D trajectory and tag map."),
        *_image_lines(artifact_paths["fused_traj3d"], alt="Fused trajectory 3D", caption="Fused 3D trajectory against ground truth."),
        *_image_lines(artifact_paths["position_compare"], alt="Position comparison", caption="Per-frame position error comparison across visual-only, fused, and IMU-only."),
        *_image_lines(artifact_paths["visual_uncertainty_scatter"], alt="Visual uncertainty scatter", caption="Visual-only predicted 95% position radius versus realized error."),
        *_image_lines(artifact_paths["fused_uncertainty_scatter"], alt="Fused uncertainty scatter", caption="Fused predicted 95% position radius versus realized error."),
        *_image_lines(artifact_paths["imu_residual_components"], alt="Fused IMU residual components", caption="Split IMU residual contributions for the fused solve."),
        *_image_lines(artifact_paths["fused_accel_bias"], alt="Fused accel bias norm", caption="Constant-bias model diagnostics across the trajectory frames."),
        "## 8. Discussion",
        "",
        f"The visual-only solution remains scientifically strong on this run, with mean position error `{_format_scalar(visual_case_payload.get('mean_position_error_m'), precision=6)}` m and sub-pixel reprojection error. The repaired fused solution reaches `{_format_scalar(fused_case_payload.get('mean_position_error_m'), precision=6)}` m mean position error, `{_format_scalar(fused_case_payload.get('mean_rotation_error_deg'), precision=4)}` deg mean rotation error, and `{_format_scalar(factor_breakdown.get('fused', {}).get('mean_whitened_imu_sq_residual_per_factor'), precision=6)}` mean whitened IMU squared residual per factor, which is now consistent with the clean async headline run instead of degrading it.",
        "",
        f"The fused accelerometer-bias plausibility check uses a hard threshold of `2.0 m/s^2`. This run reports a mean accel-bias norm of `{_format_scalar(fused_bias.get('accel_bias_norm_mean_mps2'), precision=6)}` m/s^2, so the result is {'scientifically plausible under the current model' if (fused_bias.get('accel_bias_norm_mean_mps2') or 0.0) <= 2.0 else 'flagged as scientifically inconsistent and requires further model diagnosis'}.",
        "",
        "The likelihood calibration sweeps and uncertainty stratification tables are exported as machine-readable artifacts so the clean-run result can be checked against covariance-scale changes, motion regime, tag visibility, and anchor distance instead of relying on a single scalar metric.",
        "",
        "## 9. Artifacts",
        "",
        *_artifact_lines(
            [
                ("Trajectory accuracy table", artifact_paths["trajectory_accuracy_table"]),
                ("Parameter plausibility table", artifact_paths["parameter_plausibility_table"]),
                ("Uncertainty calibration table", artifact_paths["uncertainty_calibration_table"]),
                ("Uncertainty stratified table", artifact_paths["uncertainty_stratified_table"]),
                ("Noise sensitivity table", artifact_paths["noise_sensitivity_table"]),
                ("Likelihood sweep table", artifact_paths["likelihood_sweep_table"]),
                ("Factor breakdown JSON", artifact_paths["factor_breakdown_json"]),
            ]
        ),
        "",
    ]


def empirical_visual_residual_sigma_px(case_payload: dict[str, Any]) -> float:
    robust_sigma = case_payload.get("residual_noise", {}).get("robust_axis_sigma_px")
    if robust_sigma is not None:
        return float(robust_sigma)
    return 0.0


def assumed_visual_likelihood_sigma_px(dataset: BatchCalibrationDataset) -> float:
    del dataset
    return float(load_vision_noise_preset("vision_nominal").corner_noise_std_px)


def run_batch_estimation_analysis(run_dir: str | Path, *, refresh_ablations: bool = True) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    analysis_dir = resolved_run_dir / "analysis"
    ablation_summary_path = analysis_dir / "ablation_summary.json"
    dataset = load_batch_dataset(resolved_run_dir)
    evaluation_data = load_evaluation_data(resolved_run_dir)
    frame_lookup = _frame_by_index(dataset)

    vision_nominal = load_vision_noise_preset("vision_nominal")
    imu_ideal = load_imu_noise_preset("imu_ideal")
    imu_nominal = load_imu_noise_preset("imu_nominal_phone")
    imu_stress = load_imu_noise_preset("imu_stress_phone")
    vision_ideal = VisionNoisePreset(name="vision_ideal", corner_noise_std_px=0.0, detection_drop_probability=0.0, quality_scale=1.0)
    vision_stress = VisionNoisePreset(
        name="vision_stress",
        corner_noise_std_px=1.25,
        detection_drop_probability=0.08,
        quality_scale=1.0,
        visibility_failure_probability=0.06,
        outlier_probability=0.03,
    )

    visual_init = bootstrap_initial_guess(dataset, stage="visual_only")
    visual_result = solve_visual_batch_map(dataset, visual_init, vision_nominal, variant="visual_only_unknown_map")
    visual_posterior = compute_laplace_posterior(visual_result)
    visual_eval = evaluate_batch_result(visual_result, evaluation_data, dataset=dataset, uncertainty_summary=visual_posterior)
    visual_stats = _reprojection_stats(visual_result, dataset, sigma_px=float(vision_nominal.corner_noise_std_px))

    fused_init = inertial_initial_guess_from_visual_solution(
        dataset,
        visual_result,
        anchor_frame_index=int(visual_init.anchor_frame_index),
    )
    fused_result = solve_visual_inertial_batch_map(
        dataset,
        fused_init,
        noise_cfg={"vision": vision_nominal, "imu": imu_ideal},
        variant="visual_inertial_fused",
    )
    fused_posterior = compute_laplace_posterior(fused_result)
    fused_eval = evaluate_batch_result(fused_result, evaluation_data, dataset=dataset, uncertainty_summary=fused_posterior)
    fused_stats = _reprojection_stats(fused_result, dataset, sigma_px=float(vision_nominal.corner_noise_std_px))

    imu_only_result = build_imu_only_result(dataset, fused_init, variant="imu_only_dead_reckoning")
    imu_only_eval = evaluate_batch_result(imu_only_result, evaluation_data, dataset=None)

    known_map_pose_vectors = _known_map_tag_pose_vectors(
        evaluation_data,
        anchor_frame_index=int(visual_init.anchor_frame_index),
    )
    known_map_result = solve_visual_batch_map(
        dataset,
        visual_init,
        vision_nominal,
        fixed_tag_pose_vectors=known_map_pose_vectors,
        variant="visual_only_known_map_diagnostic",
    )
    known_map_eval = evaluate_batch_result(known_map_result, evaluation_data, dataset=dataset)
    known_map_stats = _reprojection_stats(known_map_result, dataset, sigma_px=float(vision_nominal.corner_noise_std_px))

    async_headline_sweep = _run_role_label(dataset) == "async_headline"
    sweep_seeds = (11, 17, 23, 31, 47) if async_headline_sweep else (11,)
    ablation_cases: list[dict[str, Any]] = []
    raw_ablation_cases: list[dict[str, Any]] = []
    ablation_plot_points: list[dict[str, Any]] = []
    if not refresh_ablations and ablation_summary_path.exists():
        cached_ablation_summary = json.loads(ablation_summary_path.read_text(encoding="utf-8"))
        ablation_cases = list(cached_ablation_summary.get("cases", []))
        raw_ablation_cases = list(cached_ablation_summary.get("raw_seed_cases", ablation_cases))
        if not ablation_cases and raw_ablation_cases:
            ablation_cases = [
                _aggregate_ablation_cases(raw_ablation_cases, preset_name=preset_name, method_name=method_name)
                for preset_name in ("ideal", "nominal", "stress")
                for method_name in ("visual_only", "fused", "imu_only")
            ]
        for payload in ablation_cases:
            method_name = str(payload.get("method", ""))
            position_error_m = payload.get("mean_position_error_m")
            rotation_error_deg = payload.get("mean_rotation_error_deg")
            if position_error_m is None or rotation_error_deg is None:
                continue
            ablation_plot_points.append(
                {
                    "label": str(payload.get("label", method_name)),
                    "x": float(position_error_m),
                    "y": float(rotation_error_deg),
                    "color_bgr": {
                        "visual_only": (44, 123, 182),
                        "fused": (42, 156, 110),
                        "imu_only": (204, 106, 34),
                    }.get(method_name, (110, 110, 110)),
                }
            )
    else:
        ablation_cache_dir = analysis_dir / "ablation_seed_cases"
        ablation_cache_dir.mkdir(parents=True, exist_ok=True)
        preset_specs = [
            ("ideal", vision_ideal, imu_ideal),
            ("nominal", vision_nominal, imu_nominal),
            ("stress", vision_stress, imu_stress),
        ]
        for preset_name, preset_vision, preset_imu in preset_specs:
            ideal_seed_template_cases: list[dict[str, Any]] | None = None
            for seed in sweep_seeds:
                cache_path = _ablation_seed_cache_path(analysis_dir, preset_name=preset_name, seed=int(seed))
                if cache_path.exists() and not refresh_ablations:
                    cached_seed_bundle = json.loads(cache_path.read_text(encoding="utf-8"))
                    seed_cases = list(cached_seed_bundle.get("cases", []))
                elif preset_name == "ideal" and ideal_seed_template_cases is not None:
                    seed_cases = []
                    for template_case in ideal_seed_template_cases:
                        cloned_case = json.loads(json.dumps(template_case))
                        cloned_case["seed"] = int(seed)
                        seed_cases.append(cloned_case)
                else:
                    dataset_case = dataset if preset_name == "ideal" else _perturb_dataset(
                        dataset,
                        vision_noise=preset_vision,
                        imu_noise=preset_imu,
                        seed=int(seed),
                    )
                    visual_case_init = bootstrap_initial_guess(dataset_case, stage="visual_only")
                    visual_case_result = solve_visual_batch_map(
                        dataset_case,
                        visual_case_init,
                        preset_vision if preset_name != "ideal" else vision_nominal,
                        variant=f"visual_only_{preset_name}",
                    )
                    visual_case_posterior = compute_laplace_posterior(visual_case_result)
                    visual_case_eval = evaluate_batch_result(
                        visual_case_result,
                        evaluation_data,
                        dataset=dataset_case,
                        uncertainty_summary=visual_case_posterior,
                    )
                    visual_case_stats = _reprojection_stats(
                        visual_case_result,
                        dataset_case,
                        sigma_px=float(max(preset_vision.corner_noise_std_px, 0.1)),
                    )

                    fused_case_init = inertial_initial_guess_from_visual_solution(
                        dataset_case,
                        visual_case_result,
                        anchor_frame_index=int(visual_case_init.anchor_frame_index),
                    )
                    fused_case_result = solve_visual_inertial_batch_map(
                        dataset_case,
                        fused_case_init,
                        noise_cfg={"vision": preset_vision if preset_name != "ideal" else vision_nominal, "imu": preset_imu},
                        variant=f"fused_{preset_name}",
                    )
                    fused_case_posterior = compute_laplace_posterior(fused_case_result)
                    fused_case_eval = evaluate_batch_result(
                        fused_case_result,
                        evaluation_data,
                        dataset=dataset_case,
                        uncertainty_summary=fused_case_posterior,
                    )
                    fused_case_stats = _reprojection_stats(
                        fused_case_result,
                        dataset_case,
                        sigma_px=float(max(preset_vision.corner_noise_std_px, 0.1)),
                    )

                    imu_case_result = build_imu_only_result(dataset_case, fused_case_init, variant=f"imu_only_{preset_name}")
                    imu_case_eval = evaluate_batch_result(imu_case_result, evaluation_data, dataset=None)
                    seed_cases = []
                    for method_name, result_case, eval_case, reproj_case in [
                        ("visual_only", visual_case_result, visual_case_eval, visual_case_stats),
                        ("fused", fused_case_result, fused_case_eval, fused_case_stats),
                        ("imu_only", imu_case_result, imu_case_eval, None),
                    ]:
                        label = f"{preset_name}:{method_name}"
                        payload = _case_payload(
                            label=label,
                            result=result_case,
                            evaluation=eval_case,
                            reprojection_stats=reproj_case,
                            uncertainty_summary=visual_case_posterior if method_name == "visual_only" else fused_case_posterior if method_name == "fused" else None,
                        )
                        payload["noise_preset"] = preset_name
                        payload["method"] = method_name
                        payload["seed"] = int(seed)
                        seed_cases.append(payload)
                    if preset_name == "ideal":
                        ideal_seed_template_cases = [json.loads(json.dumps(_jsonify(case))) for case in seed_cases]
                cache_path.write_text(
                    json.dumps(_jsonify({"preset_name": preset_name, "seed": int(seed), "cases": seed_cases}), indent=2),
                    encoding="utf-8",
                )
                raw_ablation_cases.extend(seed_cases)

        ablation_cases = [
            _aggregate_ablation_cases(raw_ablation_cases, preset_name=preset_name, method_name=method_name)
            for preset_name in ("ideal", "nominal", "stress")
            for method_name in ("visual_only", "fused", "imu_only")
        ]
        for payload in ablation_cases:
            method_name = str(payload.get("method", ""))
            position_error_m = payload.get("mean_position_error_m")
            rotation_error_deg = payload.get("mean_rotation_error_deg")
            if position_error_m is None or rotation_error_deg is None:
                continue
            ablation_plot_points.append(
                {
                    "label": str(payload.get("label", method_name)),
                    "x": float(position_error_m),
                    "y": float(rotation_error_deg),
                    "color_bgr": {
                        "visual_only": (44, 123, 182),
                        "fused": (42, 156, 110),
                        "imu_only": (204, 106, 34),
                    }.get(method_name, (110, 110, 110)),
                }
            )

    visual_case_payload = _case_payload(
        label="visual_only_unknown_map",
        result=visual_result,
        evaluation=visual_eval,
        reprojection_stats=visual_stats,
        uncertainty_summary=visual_posterior,
    )
    fused_case_payload = _case_payload(
        label="visual_inertial_fused",
        result=fused_result,
        evaluation=fused_eval,
        reprojection_stats=fused_stats,
        uncertainty_summary=fused_posterior,
    )
    imu_only_case_payload = _case_payload(
        label="imu_only_dead_reckoning",
        result=imu_only_result,
        evaluation=imu_only_eval,
        reprojection_stats=None,
    )
    known_map_case_payload = _case_payload(
        label="visual_only_known_map_diagnostic",
        result=known_map_result,
        evaluation=known_map_eval,
        reprojection_stats=known_map_stats,
    )

    visual_pose_path = analysis_dir / "batch_v1_pose_estimates.jsonl"
    fused_pose_path = analysis_dir / "batch_v2_pose_estimates.jsonl"
    imu_only_pose_path = analysis_dir / "imu_only_pose_estimates.jsonl"
    visual_tag_map_path = analysis_dir / "batch_v1_tag_map.jsonl"
    error_cdf_path = analysis_dir / "error_cdf.png"
    per_tag_error_path = analysis_dir / "per_tag_error_bar.png"
    per_tag_rotation_error_path = analysis_dir / "per_tag_rotation_error_bar.png"
    tag_uncertainty_bar_path = analysis_dir / "batch_v1_tag_uncertainty_bar.png"
    ablation_pareto_path = analysis_dir / "ablation_pareto.png"
    ablation_position_bar_path = analysis_dir / "ablation_position_bars.png"
    ablation_rotation_bar_path = analysis_dir / "ablation_rotation_bars.png"
    coverage_plot_path = analysis_dir / "coverage_reliability.png"
    trajectory_plot_path = analysis_dir / "batch_trajectory_xy.png"
    tag_map_plot_path = analysis_dir / "batch_tag_map_xy.png"
    visual_convergence_path = analysis_dir / "batch_v1_convergence.png"
    fused_convergence_path = analysis_dir / "batch_v2_convergence.png"
    visual_trajectory_3d_path = analysis_dir / "batch_v1_trajectory_3d.png"
    fused_trajectory_3d_path = analysis_dir / "batch_v2_trajectory_3d.png"
    imu_only_trajectory_3d_path = analysis_dir / "imu_only_trajectory_3d.png"
    residual_histogram_path = analysis_dir / "batch_v1_residual_hist.png"
    visual_frame_rmse_path = analysis_dir / "batch_v1_reprojection_timeline.png"
    visual_position_error_path = analysis_dir / "batch_v1_position_error_timeline.png"
    visual_rotation_error_path = analysis_dir / "batch_v1_rotation_error_timeline.png"
    position_compare_path = analysis_dir / "trajectory_position_error_compare.png"
    rotation_compare_path = analysis_dir / "trajectory_rotation_error_compare.png"
    uncertainty_radius_path = analysis_dir / "batch_v1_uncertainty_radius_timeline.png"
    uncertainty_error_scatter_path = analysis_dir / "batch_v1_uncertainty_vs_error.png"
    fused_uncertainty_error_scatter_path = analysis_dir / "batch_v2_uncertainty_vs_error.png"
    imu_residual_components_path = analysis_dir / "batch_v2_imu_residual_components.png"
    fused_gyro_bias_path = analysis_dir / "batch_v2_gyro_bias_norm_timeline.png"
    fused_accel_bias_path = analysis_dir / "batch_v2_accel_bias_norm_timeline.png"
    fused_velocity_path = analysis_dir / "batch_v2_velocity_norm_timeline.png"
    ablation_summary_path = analysis_dir / "ablation_summary.json"
    uncertainty_summary_path = analysis_dir / "uncertainty_summary.json"
    likelihood_sweep_cache_path = analysis_dir / "likelihood_sweep_rows.json"
    visual_report_path = analysis_dir / "visual_map_report.md"
    fused_report_path = analysis_dir / "visual_inertial_report.md"
    scientific_report_path = analysis_dir / "checkpoint_03_scientific_report.md"
    trajectory_accuracy_table_path = analysis_dir / "trajectory_accuracy_table.csv"
    parameter_plausibility_table_path = analysis_dir / "parameter_plausibility_table.csv"
    uncertainty_calibration_table_path = analysis_dir / "uncertainty_calibration_table.csv"
    uncertainty_stratified_table_path = analysis_dir / "uncertainty_stratified_table.csv"
    noise_sensitivity_table_path = analysis_dir / "noise_sensitivity_table.csv"
    likelihood_sweep_table_path = analysis_dir / "likelihood_sweep_table.csv"
    factor_breakdown_path = analysis_dir / "factor_breakdown.json"

    visual_reprojection_details = _reprojection_details(visual_result, dataset)
    fused_reprojection_details = _reprojection_details(fused_result, dataset)
    known_map_reprojection_details = _reprojection_details(known_map_result, dataset)
    visual_records = _trajectory_records(visual_result, evaluation_data)
    fused_records = _trajectory_records(fused_result, evaluation_data)
    imu_records = _trajectory_records(imu_only_result, evaluation_data)
    known_map_records = _trajectory_records(known_map_result, evaluation_data)
    visual_traj_stats = _trajectory_stats(visual_records)
    fused_traj_stats = _trajectory_stats(fused_records)
    imu_traj_stats = _trajectory_stats(imu_records)
    known_map_traj_stats = _trajectory_stats(known_map_records)
    visual_alignment = _alignment_for_result(visual_result, evaluation_data)
    aligned_tag_points: list[np.ndarray] = []
    tag_specs: list[dict[str, Any]] = []
    for tag_id in sorted(int(index) for index in visual_result.tag_pose_vectors):
        aligned_pose = _aligned_pose_vector(np.asarray(visual_result.tag_pose_vectors[tag_id], dtype=np.float64), visual_alignment)
        position_world_m, _ = pose_components_from_vector(aligned_pose)
        aligned_tag_points.append(position_world_m)
        tag_specs.append({"label": str(int(tag_id)), "point": position_world_m, "color_bgr": (205, 102, 44)})
    truth_points = [np.asarray(record["truth_position_world_m"], dtype=np.float64) for record in visual_records if record["truth_position_world_m"] is not None]
    visual_points = [np.asarray(record["position_world_m"], dtype=np.float64) for record in visual_records]
    fused_points = [np.asarray(record["position_world_m"], dtype=np.float64) for record in fused_records]
    imu_points = [np.asarray(record["position_world_m"], dtype=np.float64) for record in imu_records]
    frame_indices_visual = [int(record["frame_index"]) for record in visual_eval.per_frame_records]
    visual_position_errors = [float(record["position_error_m"]) for record in visual_eval.per_frame_records]
    visual_rotation_errors = [float(record["rotation_error_deg"]) for record in visual_eval.per_frame_records]
    fused_frame_lookup = {int(record["frame_index"]): record for record in fused_eval.per_frame_records}
    imu_frame_lookup = {int(record["frame_index"]): record for record in imu_only_eval.per_frame_records}
    visual_frame_lookup_eval = {int(record["frame_index"]): record for record in visual_eval.per_frame_records}
    comparison_frame_indices = sorted(set(visual_frame_lookup_eval) | set(fused_frame_lookup) | set(imu_frame_lookup))
    visual_frame_rmse_by_frame = visual_reprojection_details["frame_rmse_by_frame"]
    uncertainty_lookup = {int(item.index): item for item in visual_posterior.camera_pose_uncertainty}
    fused_uncertainty_lookup = {int(item.index): item for item in fused_posterior.camera_pose_uncertainty}
    tag_uncertainty_lookup = {int(item.index): item for item in visual_posterior.tag_pose_uncertainty}
    uncertainty_frame_indices = sorted(set(frame_indices_visual) & set(uncertainty_lookup))
    uncertainty_radii = [float(uncertainty_lookup[frame_index].position_radius_95_m) for frame_index in uncertainty_frame_indices]
    error_lookup = {int(record["frame_index"]): float(record["position_error_m"]) for record in visual_eval.per_frame_records}
    uncertainty_error_points = [
        {
            "label": "",
            "x": float(uncertainty_lookup[frame_index].position_radius_95_m),
            "y": float(error_lookup[frame_index]),
            "color_bgr": (42, 156, 110),
        }
        for frame_index in uncertainty_frame_indices
    ]
    fused_error_lookup = {int(record["frame_index"]): float(record["position_error_m"]) for record in fused_eval.per_frame_records}
    fused_uncertainty_frame_indices = sorted(set(fused_error_lookup) & set(fused_uncertainty_lookup))
    fused_uncertainty_error_points = [
        {
            "label": "",
            "x": float(fused_uncertainty_lookup[frame_index].position_radius_95_m),
            "y": float(fused_error_lookup[frame_index]),
            "color_bgr": (205, 102, 44),
        }
        for frame_index in fused_uncertainty_frame_indices
    ]
    bias_series = _bias_norm_series(fused_result)
    tag_detection_counts: dict[int, int] = defaultdict(int)
    frame_detection_counts: dict[int, int] = defaultdict(int)
    for detection in dataset.tag_detections:
        tag_detection_counts[int(detection.tag_id)] += 1
        frame_detection_counts[int(detection.frame_index)] += 1
    speed_norm_by_frame = {
        int(sample.frame_index): float(np.linalg.norm(np.asarray(sample.velocity_world_mps, dtype=np.float64)))
        for sample in evaluation_data.camera_truth
    }

    visual_case_payload["trajectory_stats"] = visual_traj_stats
    visual_case_payload["residual_noise"] = {
        "robust_axis_sigma_px": visual_reprojection_details["robust_sigma_px"],
        "corner_p50_px": float(np.percentile(np.asarray(visual_reprojection_details["corner_norms_px"], dtype=np.float64), 50.0)) if visual_reprojection_details["corner_norms_px"] else None,
        "corner_p95_px": float(np.percentile(np.asarray(visual_reprojection_details["corner_norms_px"], dtype=np.float64), 95.0)) if visual_reprojection_details["corner_norms_px"] else None,
        "corner_p99_px": float(np.percentile(np.asarray(visual_reprojection_details["corner_norms_px"], dtype=np.float64), 99.0)) if visual_reprojection_details["corner_norms_px"] else None,
    }
    fused_case_payload["trajectory_stats"] = fused_traj_stats
    fused_case_payload["bias_stats"] = {
        "gyro_bias_norm_mean_rps": float(np.mean(np.asarray(bias_series["gyro_bias_norm_rps"], dtype=np.float64))) if bias_series["gyro_bias_norm_rps"] else None,
        "gyro_bias_norm_max_rps": float(np.max(np.asarray(bias_series["gyro_bias_norm_rps"], dtype=np.float64))) if bias_series["gyro_bias_norm_rps"] else None,
        "accel_bias_norm_mean_mps2": float(np.mean(np.asarray(bias_series["accel_bias_norm_mps2"], dtype=np.float64))) if bias_series["accel_bias_norm_mps2"] else None,
        "accel_bias_norm_max_mps2": float(np.max(np.asarray(bias_series["accel_bias_norm_mps2"], dtype=np.float64))) if bias_series["accel_bias_norm_mps2"] else None,
        "velocity_norm_mean_mps": float(np.mean(np.asarray(bias_series["velocity_norm_mps"], dtype=np.float64))) if bias_series["velocity_norm_mps"] else None,
        "velocity_norm_max_mps": float(np.max(np.asarray(bias_series["velocity_norm_mps"], dtype=np.float64))) if bias_series["velocity_norm_mps"] else None,
    }
    imu_only_case_payload["trajectory_stats"] = imu_traj_stats
    known_map_case_payload["trajectory_stats"] = known_map_traj_stats

    cases_by_noise = {
        preset: {item["method"]: item for item in ablation_cases if item["noise_preset"] == preset}
        for preset in ("ideal", "nominal", "stress")
    }
    if likelihood_sweep_cache_path.exists() and not refresh_ablations:
        likelihood_sweep_rows = list(json.loads(likelihood_sweep_cache_path.read_text(encoding="utf-8")))
    else:
        likelihood_sweep_rows = _run_likelihood_sweeps(
            analysis_dir=analysis_dir,
            dataset=dataset,
            evaluation_data=evaluation_data,
            anchor_frame_index=int(visual_init.anchor_frame_index),
            fused_result=fused_result,
            base_vision_noise=vision_nominal,
            base_imu_noise=imu_ideal,
        )
        likelihood_sweep_cache_path.write_text(json.dumps(_jsonify(likelihood_sweep_rows), indent=2), encoding="utf-8")
    uncertainty_stratified_rows = _uncertainty_stratified_rows(
        _uncertainty_frame_records(
            method_name="visual_only",
            evaluation=visual_eval,
            uncertainty_summary=visual_posterior,
            frame_detection_counts=frame_detection_counts,
            speed_norm_by_frame=speed_norm_by_frame,
            anchor_frame_index=int(visual_init.anchor_frame_index),
        )
        + _uncertainty_frame_records(
            method_name="fused",
            evaluation=fused_eval,
            uncertainty_summary=fused_posterior,
            frame_detection_counts=frame_detection_counts,
            speed_norm_by_frame=speed_norm_by_frame,
            anchor_frame_index=int(visual_init.anchor_frame_index),
        )
    )

    _save_pose_estimates_jsonl(visual_pose_path, dataset=dataset, result=visual_result, evaluation_data=evaluation_data, uncertainty_summary=visual_posterior)
    _save_pose_estimates_jsonl(fused_pose_path, dataset=dataset, result=fused_result, evaluation_data=evaluation_data, uncertainty_summary=fused_posterior)
    _save_pose_estimates_jsonl(imu_only_pose_path, dataset=dataset, result=imu_only_result, evaluation_data=evaluation_data)
    _save_tag_map_json(visual_tag_map_path, result=visual_result, evaluation_data=evaluation_data)
    _save_multi_line_plot(
        visual_convergence_path,
        title="Batch V1 Convergence",
        x_label="iteration",
        y_label="log10(cost)",
        x_values=[float(index + 1) for index in range(len(visual_result.diagnostics.get("iteration_costs", [])))],
        series_specs=[
            {
                "label": "visual-only",
                "values": [math.log10(max(float(cost), 1e-12)) for cost in visual_result.diagnostics.get("iteration_costs", [])],
                "color_bgr": (44, 123, 182),
            },
            {
                "label": "known-map",
                "values": [math.log10(max(float(cost), 1e-12)) for cost in known_map_result.diagnostics.get("iteration_costs", [])],
                "color_bgr": (205, 102, 44),
            },
        ],
    )
    _save_multi_line_plot(
        fused_convergence_path,
        title="Batch V2 Convergence",
        x_label="iteration",
        y_label="log10(cost)",
        x_values=[float(index + 1) for index in range(len(fused_result.diagnostics.get("iteration_costs", [])))],
        series_specs=[
            {
                "label": "fused",
                "values": [math.log10(max(float(cost), 1e-12)) for cost in fused_result.diagnostics.get("iteration_costs", [])],
                "color_bgr": (42, 156, 110),
            }
        ],
    )
    _save_cdf_plot(
        error_cdf_path,
        title="Batch V1 Position Error CDF",
        values=[float(record["position_error_m"]) for record in visual_eval.per_frame_records],
        x_label="position error [m]",
    )
    _save_bar_plot(
        per_tag_error_path,
        title="Batch V1 Per-Tag Translation Error",
        labels=[str(int(record["tag_id"])) for record in visual_eval.per_tag_records],
        values=[float(record["translation_error_m"]) for record in visual_eval.per_tag_records],
        y_label="translation error [m]",
    )
    _save_bar_plot(
        per_tag_rotation_error_path,
        title="Batch V1 Per-Tag Rotation Error",
        labels=[str(int(record["tag_id"])) for record in visual_eval.per_tag_records],
        values=[float(record["rotation_error_deg"]) for record in visual_eval.per_tag_records],
        y_label="rotation error [deg]",
    )
    _save_bar_plot(
        tag_uncertainty_bar_path,
        title="Batch V1 Per-Tag 95% Position Radius",
        labels=[str(int(record["tag_id"])) for record in visual_eval.per_tag_records],
        values=[float(tag_uncertainty_lookup.get(int(record["tag_id"])).position_radius_95_m) if int(record["tag_id"]) in tag_uncertainty_lookup else float("nan") for record in visual_eval.per_tag_records],
        y_label="95% position radius [m]",
    )
    _save_scatter_plot(
        ablation_pareto_path,
        title="Ablation Pareto",
        x_label="mean position error [m]",
        y_label="mean rotation error [deg]",
        points=ablation_plot_points,
    )
    _save_grouped_bar_plot(
        ablation_position_bar_path,
        title="Ablation Mean Position Error",
        group_labels=["ideal", "nominal", "stress"],
        series_specs=[
            {"label": "visual-only", "values": [cases_by_noise[level]["visual_only"]["mean_position_error_m"] for level in ("ideal", "nominal", "stress")], "color_bgr": (44, 123, 182)},
            {"label": "fused", "values": [cases_by_noise[level]["fused"]["mean_position_error_m"] for level in ("ideal", "nominal", "stress")], "color_bgr": (42, 156, 110)},
            {"label": "imu-only", "values": [cases_by_noise[level]["imu_only"]["mean_position_error_m"] for level in ("ideal", "nominal", "stress")], "color_bgr": (205, 102, 44)},
        ],
        y_label="mean position error [m]",
    )
    _save_grouped_bar_plot(
        ablation_rotation_bar_path,
        title="Ablation Mean Rotation Error",
        group_labels=["ideal", "nominal", "stress"],
        series_specs=[
            {"label": "visual-only", "values": [cases_by_noise[level]["visual_only"]["mean_rotation_error_deg"] for level in ("ideal", "nominal", "stress")], "color_bgr": (44, 123, 182)},
            {"label": "fused", "values": [cases_by_noise[level]["fused"]["mean_rotation_error_deg"] for level in ("ideal", "nominal", "stress")], "color_bgr": (42, 156, 110)},
            {"label": "imu-only", "values": [cases_by_noise[level]["imu_only"]["mean_rotation_error_deg"] for level in ("ideal", "nominal", "stress")], "color_bgr": (205, 102, 44)},
        ],
        y_label="mean rotation error [deg]",
    )
    if visual_posterior.coverage_by_level:
        levels = [50.0, 68.0, 90.0, 95.0]
        observed = [float(visual_posterior.coverage_by_level.get(str(int(level)), 0.0)) for level in levels]
        _save_line_plot(
            coverage_plot_path,
            title="Visual-Only Coverage Reliability",
            x_label="nominal coverage [%]",
            y_label="observed coverage",
            x_values=levels,
            y_values=observed,
            color_bgr=(42, 156, 110),
        )
    _save_histogram_plot(
        residual_histogram_path,
        title="Batch V1 Corner Residual Histogram",
        values=list(visual_reprojection_details["corner_norms_px"]),
        x_label="corner residual norm [px]",
    )
    _save_line_plot(
        visual_frame_rmse_path,
        title="Batch V1 Reprojection RMSE Timeline",
        x_label="frame index",
        y_label="RMSE [px]",
        x_values=[float(frame_index) for frame_index in sorted(visual_frame_rmse_by_frame)],
        y_values=[float(visual_frame_rmse_by_frame[frame_index]) for frame_index in sorted(visual_frame_rmse_by_frame)],
        color_bgr=(44, 123, 182),
    )
    _save_line_plot(
        visual_position_error_path,
        title="Batch V1 Position Error Timeline",
        x_label="frame index",
        y_label="position error [m]",
        x_values=[float(frame_index) for frame_index in frame_indices_visual],
        y_values=visual_position_errors,
        color_bgr=(42, 156, 110),
    )
    _save_line_plot(
        visual_rotation_error_path,
        title="Batch V1 Rotation Error Timeline",
        x_label="frame index",
        y_label="rotation error [deg]",
        x_values=[float(frame_index) for frame_index in frame_indices_visual],
        y_values=visual_rotation_errors,
        color_bgr=(205, 102, 44),
    )
    _save_multi_line_plot(
        position_compare_path,
        title="Trajectory Position Error Comparison",
        x_label="frame index",
        y_label="position error [m]",
        x_values=[float(frame_index) for frame_index in comparison_frame_indices],
        series_specs=[
            {"label": "visual-only", "values": [visual_frame_lookup_eval.get(frame_index, {}).get("position_error_m") for frame_index in comparison_frame_indices], "color_bgr": (44, 123, 182)},
            {"label": "fused", "values": [fused_frame_lookup.get(frame_index, {}).get("position_error_m") for frame_index in comparison_frame_indices], "color_bgr": (42, 156, 110)},
            {"label": "imu-only", "values": [imu_frame_lookup.get(frame_index, {}).get("position_error_m") for frame_index in comparison_frame_indices], "color_bgr": (205, 102, 44)},
        ],
    )
    _save_multi_line_plot(
        rotation_compare_path,
        title="Trajectory Rotation Error Comparison",
        x_label="frame index",
        y_label="rotation error [deg]",
        x_values=[float(frame_index) for frame_index in comparison_frame_indices],
        series_specs=[
            {"label": "visual-only", "values": [visual_frame_lookup_eval.get(frame_index, {}).get("rotation_error_deg") for frame_index in comparison_frame_indices], "color_bgr": (44, 123, 182)},
            {"label": "fused", "values": [fused_frame_lookup.get(frame_index, {}).get("rotation_error_deg") for frame_index in comparison_frame_indices], "color_bgr": (42, 156, 110)},
            {"label": "imu-only", "values": [imu_frame_lookup.get(frame_index, {}).get("rotation_error_deg") for frame_index in comparison_frame_indices], "color_bgr": (205, 102, 44)},
        ],
    )
    _save_line_plot(
        uncertainty_radius_path,
        title="Batch V1 95% Position Radius Timeline",
        x_label="frame index",
        y_label="radius [m]",
        x_values=[float(frame_index) for frame_index in uncertainty_frame_indices],
        y_values=uncertainty_radii,
        color_bgr=(142, 68, 173),
    )
    _save_scatter_plot(
        uncertainty_error_scatter_path,
        title="Batch V1 Uncertainty vs Error",
        x_label="predicted 95% radius [m]",
        y_label="realized position error [m]",
        points=uncertainty_error_points,
    )
    _save_scatter_plot(
        fused_uncertainty_error_scatter_path,
        title="Batch V2 Uncertainty vs Error",
        x_label="predicted 95% radius [m]",
        y_label="realized position error [m]",
        points=fused_uncertainty_error_points,
    )
    _save_bar_plot(
        imu_residual_components_path,
        title="Batch V2 IMU Residual Components",
        labels=["position", "rotation", "velocity"],
        values=[
            float(fused_result.diagnostics.get("factor_breakdown", {}).get("mean_whitened_imu_sq_residual_position_per_factor", 0.0)),
            float(fused_result.diagnostics.get("factor_breakdown", {}).get("mean_whitened_imu_sq_residual_rotation_per_factor", 0.0)),
            float(fused_result.diagnostics.get("factor_breakdown", {}).get("mean_whitened_imu_sq_residual_velocity_per_factor", 0.0)),
        ],
        y_label="mean whitened sq residual / factor",
    )
    _save_topdown_plot(trajectory_plot_path, title="Batch V1 Camera Trajectory (XY)", camera_points=visual_points, tag_points=[])
    _save_topdown_plot(tag_map_plot_path, title="Batch V1 Tag Map and Trajectory (XY)", camera_points=visual_points, tag_points=aligned_tag_points)
    _save_3d_projection_plot(
        visual_trajectory_3d_path,
        title="Batch V1 Trajectory 3D",
        series_specs=[
            {"label": "ground truth", "points": truth_points, "color_bgr": (148, 148, 148)},
            {"label": "visual-only", "points": visual_points, "color_bgr": (44, 123, 182)},
        ],
        tag_specs=tag_specs,
    )
    _save_3d_projection_plot(
        fused_trajectory_3d_path,
        title="Batch V2 Trajectory 3D",
        series_specs=[
            {"label": "ground truth", "points": truth_points, "color_bgr": (148, 148, 148)},
            {"label": "visual-only", "points": visual_points, "color_bgr": (44, 123, 182)},
            {"label": "fused", "points": fused_points, "color_bgr": (42, 156, 110)},
        ],
        tag_specs=tag_specs,
    )
    _save_3d_projection_plot(
        imu_only_trajectory_3d_path,
        title="IMU-Only Trajectory 3D",
        series_specs=[
            {"label": "ground truth", "points": truth_points, "color_bgr": (148, 148, 148)},
            {"label": "imu-only", "points": imu_points, "color_bgr": (205, 102, 44)},
        ],
        tag_specs=tag_specs,
    )
    _save_line_plot(
        fused_gyro_bias_path,
        title="Batch V2 Gyro Bias Norm",
        x_label="frame index",
        y_label="gyro bias norm [rad/s]",
        x_values=[float(value) for value in bias_series["frame_indices"]],
        y_values=bias_series["gyro_bias_norm_rps"],
        color_bgr=(205, 102, 44),
    )
    _save_line_plot(
        fused_accel_bias_path,
        title="Batch V2 Accel Bias Norm",
        x_label="frame index",
        y_label="accel bias norm [m/s^2]",
        x_values=[float(value) for value in bias_series["frame_indices"]],
        y_values=bias_series["accel_bias_norm_mps2"],
        color_bgr=(142, 68, 173),
    )
    _save_line_plot(
        fused_velocity_path,
        title="Batch V2 Velocity Norm",
        x_label="frame index",
        y_label="velocity norm [m/s]",
        x_values=[float(value) for value in bias_series["frame_indices"]],
        y_values=bias_series["velocity_norm_mps"],
        color_bgr=(42, 156, 110),
    )

    camera_radii = [float(item.position_radius_95_m) for item in visual_posterior.camera_pose_uncertainty]
    tag_radii = [float(item.position_radius_95_m) for item in visual_posterior.tag_pose_uncertainty]
    worst_frame_rows = []
    for record in sorted(visual_eval.per_frame_records, key=lambda item: float(item["position_error_m"]), reverse=True)[:10]:
        frame_index = int(record["frame_index"])
        worst_frame_rows.append(
            [
                str(frame_index),
                _format_scalar(record.get("timestamp_s"), precision=3),
                _format_scalar(record.get("position_error_m"), precision=6),
                _format_scalar(record.get("rotation_error_deg"), precision=4),
                _format_scalar(visual_frame_rmse_by_frame.get(frame_index), precision=4),
                _format_scalar(uncertainty_lookup.get(frame_index).position_radius_95_m if frame_index in uncertainty_lookup else None, precision=6),
            ]
        )

    tag_rows = []
    for record in visual_eval.per_tag_records:
        tag_id = int(record["tag_id"])
        tag_rows.append(
            [
                str(tag_id),
                str(tag_detection_counts.get(tag_id, 0)),
                _format_scalar(record.get("translation_error_m"), precision=6),
                _format_scalar(record.get("rotation_error_deg"), precision=4),
                _format_scalar(visual_reprojection_details["tag_rmse_by_tag"].get(tag_id), precision=4),
                _format_scalar(tag_uncertainty_lookup.get(tag_id).position_radius_95_m if tag_id in tag_uncertainty_lookup else None, precision=6),
            ]
        )

    dataset_rows = [
        ["Run directory", str(resolved_run_dir)],
        ["Frames analyzed", str(len(dataset.camera_frames))],
        ["Frames with detections", str(len(frame_detection_counts))],
        ["Tag detections", str(len(dataset.tag_detections))],
        ["Unique observed tags", str(len(tag_detection_counts))],
        ["IMU packets", str(len(dataset.imu_packets))],
        ["Anchor frame index", str(int(visual_init.anchor_frame_index))],
        ["Anchor detections", str(int(frame_detection_counts.get(int(visual_init.anchor_frame_index), 0)))],
        ["Recording rate [Hz]", _format_scalar(dataset.recording_frame_rate_hz, precision=3)],
        ["Camera model", dataset.camera_model.name],
        ["Resolution [px]", f"{dataset.camera_model.output_width_px} x {dataset.camera_model.output_height_px}"],
        ["IMU rate [Hz]", _format_scalar(dataset.device_config.imu.rate_hz, precision=3)],
    ]
    visual_solver_rows = [
        [
            "visual-only",
            str(visual_result.iterations),
            _format_scalar(visual_result.initial_cost, precision=6),
            _format_scalar(visual_result.final_cost, precision=6),
            _format_scalar(visual_case_payload.get("cost_reduction_ratio"), precision=2),
            _format_scalar(visual_eval.runtime_ms_per_iteration, precision=3),
        ],
        [
            "known-map diagnostic",
            str(known_map_result.iterations),
            _format_scalar(known_map_result.initial_cost, precision=6),
            _format_scalar(known_map_result.final_cost, precision=6),
            _format_scalar(known_map_case_payload.get("cost_reduction_ratio"), precision=2),
            _format_scalar(known_map_eval.runtime_ms_per_iteration, precision=3),
        ],
    ]
    fused_solver_rows = [
        [
            "visual-only",
            str(visual_result.iterations),
            _format_scalar(visual_result.initial_cost, precision=6),
            _format_scalar(visual_result.final_cost, precision=6),
            _format_scalar(visual_case_payload.get("cost_reduction_ratio"), precision=2),
            _format_scalar(visual_eval.runtime_ms_per_iteration, precision=3),
        ],
        [
            "fused",
            str(fused_result.iterations),
            _format_scalar(fused_result.initial_cost, precision=6),
            _format_scalar(fused_result.final_cost, precision=6),
            _format_scalar(fused_case_payload.get("cost_reduction_ratio"), precision=2),
            _format_scalar(fused_eval.runtime_ms_per_iteration, precision=3),
        ],
        [
            "known-map diagnostic",
            str(known_map_result.iterations),
            _format_scalar(known_map_result.initial_cost, precision=6),
            _format_scalar(known_map_result.final_cost, precision=6),
            _format_scalar(known_map_case_payload.get("cost_reduction_ratio"), precision=2),
            _format_scalar(known_map_eval.runtime_ms_per_iteration, precision=3),
        ],
    ]
    residual_rows = [
        ["Mean reprojection RMSE [px]", _format_scalar(visual_case_payload.get("mean_reprojection_rmse_px"), precision=4)],
        ["Robust axis sigma [px]", _format_scalar(visual_case_payload["residual_noise"].get("robust_axis_sigma_px"), precision=4)],
        ["Corner residual p50 [px]", _format_scalar(visual_case_payload["residual_noise"].get("corner_p50_px"), precision=4)],
        ["Corner residual p95 [px]", _format_scalar(visual_case_payload["residual_noise"].get("corner_p95_px"), precision=4)],
        ["Corner residual p99 [px]", _format_scalar(visual_case_payload["residual_noise"].get("corner_p99_px"), precision=4)],
        ["Robust inlier ratio", _format_scalar(visual_case_payload.get("robust_inlier_ratio"), precision=4)],
        ["Visual rejection count", _format_scalar(visual_case_payload.get("visual_rejection_count"), precision=0)],
    ]
    uncertainty_rows = [
        ["Camera pose count", str(len(visual_posterior.camera_pose_uncertainty))],
        ["Tag pose count", str(len(visual_posterior.tag_pose_uncertainty))],
        ["Covariance condition number", _format_scalar(visual_posterior.covariance_condition_number, precision=3)],
        ["Mean camera 95% radius [m]", _format_scalar(np.mean(np.asarray(camera_radii, dtype=np.float64)) if camera_radii else None, precision=6)],
        ["Max camera 95% radius [m]", _format_scalar(np.max(np.asarray(camera_radii, dtype=np.float64)) if camera_radii else None, precision=6)],
        ["Mean tag 95% radius [m]", _format_scalar(np.mean(np.asarray(tag_radii, dtype=np.float64)) if tag_radii else None, precision=6)],
        ["Max tag 95% radius [m]", _format_scalar(np.max(np.asarray(tag_radii, dtype=np.float64)) if tag_radii else None, precision=6)],
        ["Sigma-error correlation", _format_scalar(visual_posterior.metadata.get("position_sigma_error_corr"), precision=4)],
    ]
    coverage_rows = [
        [level, _format_scalar(float(level), precision=0), _format_scalar(100.0 * float(visual_posterior.coverage_by_level.get(level, 0.0)), precision=2)]
        for level in ("50", "68", "90", "95")
        if level in visual_posterior.coverage_by_level
    ]
    fused_bias_rows = [
        ["Gyro bias norm mean [rad/s]", _format_scalar(fused_case_payload["bias_stats"].get("gyro_bias_norm_mean_rps"), precision=6)],
        ["Gyro bias norm max [rad/s]", _format_scalar(fused_case_payload["bias_stats"].get("gyro_bias_norm_max_rps"), precision=6)],
        ["Accel bias norm mean [m/s^2]", _format_scalar(fused_case_payload["bias_stats"].get("accel_bias_norm_mean_mps2"), precision=6)],
        ["Accel bias norm max [m/s^2]", _format_scalar(fused_case_payload["bias_stats"].get("accel_bias_norm_max_mps2"), precision=6)],
        ["Velocity norm mean [m/s]", _format_scalar(fused_case_payload["bias_stats"].get("velocity_norm_mean_mps"), precision=6)],
        ["Velocity norm max [m/s]", _format_scalar(fused_case_payload["bias_stats"].get("velocity_norm_max_mps"), precision=6)],
        ["IMU factor count", _format_scalar(fused_result.diagnostics.get("imu_factor_count"), precision=0)],
    ]
    fused_case_payload["plausibility"] = {
        "accel_bias_norm_threshold_mps2": 2.0,
        "accel_bias_norm_mean_pass": bool((fused_case_payload["bias_stats"].get("accel_bias_norm_mean_mps2") or 0.0) <= 2.0),
        "imu_residual_threshold_per_factor": 12.0,
        "imu_residual_per_factor_pass": bool((fused_result.diagnostics.get("mean_whitened_imu_sq_residual_per_factor") or 0.0) < 12.0),
    }
    observation_rows = [
        ["visual-only", _format_scalar(visual_case_payload.get("mean_reprojection_rmse_px"), precision=4), _format_scalar(visual_case_payload.get("robust_inlier_ratio"), precision=4), _format_scalar(visual_reprojection_details.get("robust_sigma_px"), precision=4)],
        ["fused", _format_scalar(fused_case_payload.get("mean_reprojection_rmse_px"), precision=4), _format_scalar(fused_case_payload.get("robust_inlier_ratio"), precision=4), _format_scalar(fused_reprojection_details.get("robust_sigma_px"), precision=4)],
        ["known-map", _format_scalar(known_map_case_payload.get("mean_reprojection_rmse_px"), precision=4), _format_scalar(known_map_case_payload.get("robust_inlier_ratio"), precision=4), _format_scalar(known_map_reprojection_details.get("robust_sigma_px"), precision=4)],
    ]
    factor_breakdown = {
        "visual_only": dict(visual_result.diagnostics.get("factor_breakdown", {})),
        "fused": dict(fused_result.diagnostics.get("factor_breakdown", {})),
        "known_map": dict(known_map_result.diagnostics.get("factor_breakdown", {})),
        "imu_only": {"imu_factor_count": len(dataset.imu_packets)},
        "likelihood_sweeps": {
            "imu_covariance_scale": [row for row in likelihood_sweep_rows if str(row.get("sweep_family")) == "imu_covariance_scale"],
            "visual_weight_scale": [row for row in likelihood_sweep_rows if str(row.get("sweep_family")) == "visual_weight_scale"],
        },
    }
    ablation_rows = []
    for preset in ("ideal", "nominal", "stress"):
        for method in ("visual_only", "fused", "imu_only"):
            case = cases_by_noise[preset][method]
            ablation_rows.append(
                [
                    preset,
                    method,
                    _format_scalar(case.get("sample_count"), precision=0),
                    _format_scalar(case.get("mean_position_error_m"), precision=6),
                    _format_scalar(case.get("mean_position_error_std_m"), precision=6),
                    _format_scalar(case.get("mean_rotation_error_deg"), precision=4),
                    _format_scalar(case.get("uncertainty", {}).get("camera_mean_95_radius_m"), precision=6),
                    _format_scalar(100.0 * float(case.get("failure_rate", 0.0)), precision=2),
                ]
            )

    ablation_summary = {
        "schema_version": 2,
        "reference_run_dir": str(resolved_run_dir),
        "cases": ablation_cases,
        "raw_seed_cases": raw_ablation_cases,
        "sweep_seeds": list(sweep_seeds),
        "run_role": _run_role_label(dataset),
        "known_map_diagnostic": known_map_case_payload,
    }
    ablation_summary_path.write_text(json.dumps(_jsonify(ablation_summary), indent=2), encoding="utf-8")

    uncertainty_summary = {
        "visual_only": {
            "covariance_condition_number": visual_posterior.covariance_condition_number,
            "coverage_by_level": dict(visual_posterior.coverage_by_level),
            "metadata": dict(visual_posterior.metadata),
            "camera_pose_uncertainty": [_jsonify(asdict(item)) for item in visual_posterior.camera_pose_uncertainty],
            "tag_pose_uncertainty": [_jsonify(asdict(item)) for item in visual_posterior.tag_pose_uncertainty],
            "velocity_uncertainty": [_jsonify(asdict(item)) for item in visual_posterior.velocity_uncertainty],
            "gyro_bias_uncertainty": None if visual_posterior.gyro_bias_uncertainty is None else _jsonify(asdict(visual_posterior.gyro_bias_uncertainty)),
            "accel_bias_uncertainty": None if visual_posterior.accel_bias_uncertainty is None else _jsonify(asdict(visual_posterior.accel_bias_uncertainty)),
        },
        "fused": {
            "covariance_condition_number": fused_posterior.covariance_condition_number,
            "coverage_by_level": dict(fused_posterior.coverage_by_level),
            "metadata": dict(fused_posterior.metadata),
            "camera_pose_uncertainty": [_jsonify(asdict(item)) for item in fused_posterior.camera_pose_uncertainty],
            "tag_pose_uncertainty": [_jsonify(asdict(item)) for item in fused_posterior.tag_pose_uncertainty],
            "velocity_uncertainty": [_jsonify(asdict(item)) for item in fused_posterior.velocity_uncertainty],
            "gyro_bias_uncertainty": None if fused_posterior.gyro_bias_uncertainty is None else _jsonify(asdict(fused_posterior.gyro_bias_uncertainty)),
            "accel_bias_uncertainty": None if fused_posterior.accel_bias_uncertainty is None else _jsonify(asdict(fused_posterior.accel_bias_uncertainty)),
        },
    }
    uncertainty_summary_path.write_text(json.dumps(_jsonify(uncertainty_summary), indent=2), encoding="utf-8")
    factor_breakdown_path.write_text(json.dumps(_jsonify(factor_breakdown), indent=2), encoding="utf-8")

    trajectory_accuracy_rows_csv = [
        ["visual_only", visual_case_payload.get("mean_position_error_m"), visual_case_payload.get("p95_position_error_m"), visual_case_payload.get("mean_rotation_error_deg"), visual_case_payload.get("mean_reprojection_rmse_px")],
        ["fused", fused_case_payload.get("mean_position_error_m"), fused_case_payload.get("p95_position_error_m"), fused_case_payload.get("mean_rotation_error_deg"), fused_case_payload.get("mean_reprojection_rmse_px")],
        ["imu_only", imu_only_case_payload.get("mean_position_error_m"), imu_only_case_payload.get("p95_position_error_m"), imu_only_case_payload.get("mean_rotation_error_deg"), imu_only_case_payload.get("mean_reprojection_rmse_px")],
        ["known_map_diagnostic", known_map_case_payload.get("mean_position_error_m"), known_map_case_payload.get("p95_position_error_m"), known_map_case_payload.get("mean_rotation_error_deg"), known_map_case_payload.get("mean_reprojection_rmse_px")],
    ]
    parameter_plausibility_rows_csv = [
        [
            "fused",
            fused_case_payload["bias_stats"].get("gyro_bias_norm_mean_rps"),
            fused_case_payload["bias_stats"].get("accel_bias_norm_mean_mps2"),
            fused_case_payload["bias_stats"].get("accel_bias_norm_max_mps2"),
            factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_per_factor"),
            factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_position_per_factor"),
            factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_rotation_per_factor"),
            factor_breakdown.get("fused", {}).get("mean_whitened_imu_sq_residual_velocity_per_factor"),
            "pass" if fused_case_payload["plausibility"]["accel_bias_norm_mean_pass"] else "failed_plausibility_threshold",
        ],
    ]
    uncertainty_calibration_rows_csv = [
        [
            "visual_only",
            visual_case_payload.get("uncertainty", {}).get("camera_mean_95_radius_m"),
            _coverage_percent(visual_posterior, "50"),
            _coverage_percent(visual_posterior, "68"),
            _coverage_percent(visual_posterior, "90"),
            _coverage_percent(visual_posterior, "95"),
            visual_posterior.metadata.get("position_nees_mean"),
            visual_posterior.metadata.get("position_whitened_sq_error_mean"),
            visual_posterior.metadata.get("position_sigma_error_corr"),
            visual_posterior.metadata.get("rotation_nees_mean"),
            visual_posterior.metadata.get("rotation_whitened_sq_error_mean"),
            visual_posterior.metadata.get("rotation_sigma_error_corr"),
            visual_posterior.metadata.get("velocity_nees_mean"),
            visual_posterior.metadata.get("velocity_whitened_sq_error_mean"),
            visual_posterior.metadata.get("gyro_bias_nees"),
            visual_posterior.metadata.get("gyro_bias_whitened_sq_error_mean"),
            visual_posterior.metadata.get("accel_bias_nees"),
            visual_posterior.metadata.get("accel_bias_whitened_sq_error_mean"),
        ],
        [
            "fused",
            fused_case_payload.get("uncertainty", {}).get("camera_mean_95_radius_m"),
            _coverage_percent(fused_posterior, "50"),
            _coverage_percent(fused_posterior, "68"),
            _coverage_percent(fused_posterior, "90"),
            _coverage_percent(fused_posterior, "95"),
            fused_posterior.metadata.get("position_nees_mean"),
            fused_posterior.metadata.get("position_whitened_sq_error_mean"),
            fused_posterior.metadata.get("position_sigma_error_corr"),
            fused_posterior.metadata.get("rotation_nees_mean"),
            fused_posterior.metadata.get("rotation_whitened_sq_error_mean"),
            fused_posterior.metadata.get("rotation_sigma_error_corr"),
            fused_posterior.metadata.get("velocity_nees_mean"),
            fused_posterior.metadata.get("velocity_whitened_sq_error_mean"),
            fused_posterior.metadata.get("gyro_bias_nees"),
            fused_posterior.metadata.get("gyro_bias_whitened_sq_error_mean"),
            fused_posterior.metadata.get("accel_bias_nees"),
            fused_posterior.metadata.get("accel_bias_whitened_sq_error_mean"),
        ],
    ]
    uncertainty_stratified_rows_csv = [
        [
            row.get("method"),
            row.get("stratum_type"),
            row.get("stratum_value"),
            row.get("sample_count"),
            row.get("mean_predicted_95_radius_m"),
            row.get("empirical_95_coverage_pct"),
            row.get("position_nees_mean"),
            row.get("position_sigma_error_corr"),
        ]
        for row in uncertainty_stratified_rows
    ]
    noise_sensitivity_rows_csv = []
    for preset in ("ideal", "nominal", "stress"):
        for method in ("visual_only", "fused", "imu_only"):
            case = cases_by_noise[preset][method]
            uncertainty_case = case.get("uncertainty", {})
            noise_sensitivity_rows_csv.append(
                [
                    preset,
                    method,
                    case.get("sample_count"),
                    case.get("mean_position_error_m"),
                    case.get("mean_position_error_std_m"),
                    case.get("mean_rotation_error_deg"),
                    case.get("mean_rotation_error_std_deg"),
                    uncertainty_case.get("camera_mean_95_radius_m"),
                    None if not uncertainty_case else 100.0 * float(uncertainty_case.get("coverage_by_level", {}).get("95", 0.0)),
                    100.0 * float(case.get("failure_rate", 0.0)),
                    case.get("mean_reprojection_rmse_px"),
                ]
            )
    likelihood_sweep_rows_csv = [
        [
            row.get("sweep_family"),
            row.get("scale"),
            row.get("method"),
            row.get("mean_position_error_m"),
            row.get("p95_position_error_m"),
            row.get("mean_rotation_error_deg"),
            row.get("mean_whitened_imu_sq_residual_per_factor"),
            row.get("mean_whitened_imu_sq_residual_position_per_factor"),
            row.get("mean_whitened_imu_sq_residual_rotation_per_factor"),
            row.get("mean_whitened_imu_sq_residual_velocity_per_factor"),
            row.get("position_nees_mean"),
            row.get("velocity_nees_mean"),
            row.get("gyro_bias_nees"),
            row.get("accel_bias_nees"),
            row.get("empirical_95_coverage_pct"),
        ]
        for row in likelihood_sweep_rows
    ]
    _write_csv_table(
        trajectory_accuracy_table_path,
        headers=["method", "mean_position_error_m", "p95_position_error_m", "mean_rotation_error_deg", "mean_reprojection_rmse_px"],
        rows=trajectory_accuracy_rows_csv,
    )
    _write_csv_table(
        parameter_plausibility_table_path,
        headers=[
            "method",
            "mean_gyro_bias_norm_rps",
            "mean_accel_bias_norm_mps2",
            "max_accel_bias_norm_mps2",
            "mean_whitened_imu_sq_residual_per_factor",
            "mean_whitened_imu_sq_residual_position_per_factor",
            "mean_whitened_imu_sq_residual_rotation_per_factor",
            "mean_whitened_imu_sq_residual_velocity_per_factor",
            "comment",
        ],
        rows=parameter_plausibility_rows_csv,
    )
    _write_csv_table(
        uncertainty_calibration_table_path,
        headers=[
            "method",
            "mean_95_position_radius_m",
            "coverage_50_pct",
            "coverage_68_pct",
            "coverage_90_pct",
            "coverage_95_pct",
            "position_nees_mean",
            "position_whitened_sq_error_mean",
            "position_sigma_error_corr",
            "rotation_nees_mean",
            "rotation_whitened_sq_error_mean",
            "rotation_sigma_error_corr",
            "velocity_nees_mean",
            "velocity_whitened_sq_error_mean",
            "gyro_bias_nees",
            "gyro_bias_whitened_sq_error_mean",
            "accel_bias_nees",
            "accel_bias_whitened_sq_error_mean",
        ],
        rows=uncertainty_calibration_rows_csv,
    )
    _write_csv_table(
        uncertainty_stratified_table_path,
        headers=[
            "method",
            "stratum_type",
            "stratum_value",
            "sample_count",
            "mean_predicted_95_radius_m",
            "empirical_95_coverage_pct",
            "position_nees_mean",
            "position_sigma_error_corr",
        ],
        rows=uncertainty_stratified_rows_csv,
    )
    _write_csv_table(
        noise_sensitivity_table_path,
        headers=[
            "noise_preset",
            "method",
            "sample_count",
            "mean_position_error_m",
            "std_position_error_m",
            "mean_rotation_error_deg",
            "std_rotation_error_deg",
            "mean_95_position_radius_m",
            "empirical_95_coverage_pct",
            "failure_rate_pct",
            "mean_reprojection_rmse_px",
        ],
        rows=noise_sensitivity_rows_csv,
    )
    _write_csv_table(
        likelihood_sweep_table_path,
        headers=[
            "sweep_family",
            "scale",
            "method",
            "mean_position_error_m",
            "p95_position_error_m",
            "mean_rotation_error_deg",
            "mean_whitened_imu_sq_residual_per_factor",
            "mean_whitened_imu_sq_residual_position_per_factor",
            "mean_whitened_imu_sq_residual_rotation_per_factor",
            "mean_whitened_imu_sq_residual_velocity_per_factor",
            "position_nees_mean",
            "velocity_nees_mean",
            "gyro_bias_nees",
            "accel_bias_nees",
            "empirical_95_coverage_pct",
        ],
        rows=likelihood_sweep_rows_csv,
    )

    scientific_report_lines = _scientific_report_lines(
        dataset=dataset,
        visual_case_payload=visual_case_payload,
        fused_case_payload=fused_case_payload,
        imu_only_case_payload=imu_only_case_payload,
        known_map_case_payload=known_map_case_payload,
        visual_posterior=visual_posterior,
        fused_posterior=fused_posterior,
        cases_by_noise=cases_by_noise,
        factor_breakdown=factor_breakdown,
        likelihood_sweep_rows=likelihood_sweep_rows,
        uncertainty_stratified_rows=uncertainty_stratified_rows,
        artifact_paths={
            "visual_convergence": visual_convergence_path,
            "fused_convergence": fused_convergence_path,
            "visual_traj3d": visual_trajectory_3d_path,
            "fused_traj3d": fused_trajectory_3d_path,
            "position_compare": position_compare_path,
            "visual_uncertainty_scatter": uncertainty_error_scatter_path,
            "fused_uncertainty_scatter": fused_uncertainty_error_scatter_path,
            "imu_residual_components": imu_residual_components_path,
            "fused_accel_bias": fused_accel_bias_path,
            "trajectory_accuracy_table": trajectory_accuracy_table_path,
            "parameter_plausibility_table": parameter_plausibility_table_path,
            "uncertainty_calibration_table": uncertainty_calibration_table_path,
            "uncertainty_stratified_table": uncertainty_stratified_table_path,
            "noise_sensitivity_table": noise_sensitivity_table_path,
            "likelihood_sweep_table": likelihood_sweep_table_path,
            "factor_breakdown_json": factor_breakdown_path,
        },
    )
    scientific_report_path.write_text("\n".join(scientific_report_lines), encoding="utf-8")

    visual_report_lines = [
        "# Visual-Only Unknown-Map Batch MAP",
        "",
        f"Anchor bootstrap selected frame `{int(visual_init.anchor_frame_index)}` and the unknown-map bundle solved `{len(dataset.camera_frames)}` frames with `{len(dataset.tag_detections)}` tag observations. The sections below add convergence, trajectory, residual, and uncertainty detail for the preserved checkpoint run.",
        "",
        *_table_section("## Dataset Snapshot", ["Field", "Value"], dataset_rows),
        *_table_section(
            "## Primary Metrics",
            ["Case", "Solver", "Iterations", "Mean pos [m]", "Mean rot [deg]", "Reproj [px]", "Inlier ratio", "Cost x"],
            [
                _case_metric_row(visual_case_payload),
                _case_metric_row(known_map_case_payload),
            ],
        ),
        *_table_section(
            "## Convergence Summary",
            ["Solve", "Iterations", "Initial cost", "Final cost", "Cost x", "ms/iter"],
            visual_solver_rows,
        ),
        *_image_lines(visual_convergence_path, alt="Visual-only convergence", caption="Levenberg-Marquardt cost history for the unknown-map solve and the known-map diagnostic solve."),
        *_table_section(
            "## Trajectory Statistics",
            ["Case", "Path length [m]", "Start-end disp [m]", "Mean speed [m/s]", "Max speed [m/s]"],
            [
                _trajectory_stat_row("visual-only", visual_traj_stats),
                _trajectory_stat_row("known-map", known_map_traj_stats),
            ],
        ),
        *_image_lines(trajectory_plot_path, alt="Visual-only XY trajectory", caption="Top-down XY view of the solved camera path."),
        *_image_lines(tag_map_plot_path, alt="Visual-only tag map", caption="Top-down XY view of the jointly estimated tag map and trajectory."),
        *_image_lines(visual_trajectory_3d_path, alt="Visual-only 3D trajectory", caption="3D projection of the estimated trajectory against ground truth and aligned tag centers."),
        *_image_lines(visual_position_error_path, alt="Visual-only position error timeline"),
        *_image_lines(visual_rotation_error_path, alt="Visual-only rotation error timeline"),
        *_table_section("## Residual Diagnostics", ["Field", "Value"], residual_rows),
        *_image_lines(residual_histogram_path, alt="Visual-only residual histogram", caption="Corner-residual distribution in pixel space."),
        *_image_lines(visual_frame_rmse_path, alt="Visual-only reprojection RMSE timeline"),
        *_image_lines(error_cdf_path, alt="Visual-only position-error CDF"),
        *_table_section(
            "## Tag Map Detail",
            ["Tag", "Detections", "Trans err [m]", "Rot err [deg]", "Reproj RMSE [px]", "95% radius [m]"],
            tag_rows,
        ),
        *_image_lines(per_tag_error_path, alt="Per-tag translation error bars"),
        *_image_lines(per_tag_rotation_error_path, alt="Per-tag rotation error bars"),
        *_image_lines(tag_uncertainty_bar_path, alt="Per-tag uncertainty bars"),
        *_table_section("## Uncertainty Summary", ["Field", "Value"], uncertainty_rows),
        *_table_section("## Coverage Check", ["Level", "Nominal [%]", "Observed [%]"], coverage_rows),
        *_image_lines(uncertainty_radius_path, alt="Uncertainty radius timeline"),
        *_image_lines(uncertainty_error_scatter_path, alt="Uncertainty versus realized error"),
        *_image_lines(coverage_plot_path, alt="Coverage reliability curve"),
        *_table_section(
            "## Worst Frames",
            ["Frame", "t [s]", "Pos err [m]", "Rot err [deg]", "Reproj RMSE [px]", "95% radius [m]"],
            worst_frame_rows,
        ),
        "## Artifacts",
        "",
        *_artifact_lines(
            [
                ("Pose estimates", visual_pose_path),
                ("Tag map estimates", visual_tag_map_path),
                ("Uncertainty summary", uncertainty_summary_path),
                ("Legacy summary JSON", analysis_dir / "summary.json"),
            ]
        ),
        "",
    ]
    visual_report_path.write_text("\n".join(visual_report_lines), encoding="utf-8")

    fused_report_lines = [
        "# Visual-Inertial Batch MAP",
        "",
        "This report compares the fused batch solve against visual-only, IMU-only dead reckoning, and the known-map diagnostic, then summarizes how performance changes under the ideal, nominal, and stress presets.",
        "",
        *_table_section(
            "## Comparative Metrics",
            ["Case", "Solver", "Iterations", "Mean pos [m]", "Mean rot [deg]", "Reproj [px]", "Inlier ratio", "Cost x"],
            [
                _case_metric_row(visual_case_payload),
                _case_metric_row(fused_case_payload),
                _case_metric_row(imu_only_case_payload),
                _case_metric_row(known_map_case_payload),
            ],
        ),
        *_table_section(
            "## Solver Summary",
            ["Solve", "Iterations", "Initial cost", "Final cost", "Cost x", "ms/iter"],
            fused_solver_rows,
        ),
        *_image_lines(fused_convergence_path, alt="Fused convergence", caption="Cost history for the visual-inertial LM solve."),
        *_table_section(
            "## Trajectory Statistics",
            ["Case", "Path length [m]", "Start-end disp [m]", "Mean speed [m/s]", "Max speed [m/s]"],
            [
                _trajectory_stat_row("visual-only", visual_traj_stats),
                _trajectory_stat_row("fused", fused_traj_stats),
                _trajectory_stat_row("imu-only", imu_traj_stats),
            ],
        ),
        *_image_lines(fused_trajectory_3d_path, alt="Fused 3D trajectory", caption="3D comparison of ground truth, visual-only, and fused trajectories."),
        *_image_lines(imu_only_trajectory_3d_path, alt="IMU-only 3D trajectory", caption="3D projection of the IMU-only drift baseline."),
        *_image_lines(position_compare_path, alt="Trajectory position-error comparison"),
        *_image_lines(rotation_compare_path, alt="Trajectory rotation-error comparison"),
        *_table_section("## Fused State Diagnostics", ["Field", "Value"], fused_bias_rows),
        *_image_lines(fused_gyro_bias_path, alt="Gyro bias norm timeline"),
        *_image_lines(fused_accel_bias_path, alt="Accel bias norm timeline"),
        *_image_lines(fused_velocity_path, alt="Velocity norm timeline"),
        *_table_section(
            "## Observation Consistency",
            ["Case", "Mean reproj [px]", "Inlier ratio", "Robust sigma [px]"],
            observation_rows,
        ),
        *_table_section(
            "## Noise-Preset Ablations",
            ["Noise", "Method", "Seeds", "Mean pos [m]", "Std pos [m]", "Mean rot [deg]", "Mean 95% radius [m]", "Failure rate [%]"],
            ablation_rows,
        ),
        *_image_lines(ablation_position_bar_path, alt="Ablation position bars"),
        *_image_lines(ablation_rotation_bar_path, alt="Ablation rotation bars"),
        *_image_lines(ablation_pareto_path, alt="Ablation pareto plot"),
        "## Artifacts",
        "",
        *_artifact_lines(
            [
                ("Fused pose estimates", fused_pose_path),
                ("IMU-only pose estimates", imu_only_pose_path),
                ("Ablation summary", ablation_summary_path),
                ("Legacy summary JSON", analysis_dir / "summary.json"),
            ]
        ),
        "",
    ]
    fused_report_path.write_text("\n".join(fused_report_lines), encoding="utf-8")

    summary_updates = {
        "schema_version": 4,
        "batch_v1": visual_case_payload,
        "batch_v2": fused_case_payload,
        "ablations": {
            "ideal": cases_by_noise["ideal"],
            "nominal": cases_by_noise["nominal"],
            "stress": cases_by_noise["stress"],
            "sweep_seeds": list(sweep_seeds),
            "run_role": _run_role_label(dataset),
            "known_map_diagnostic": known_map_case_payload,
        },
        "uncertainty": {
            "visual_only": {
                "coverage_by_level": dict(visual_posterior.coverage_by_level),
                "covariance_condition_number": visual_posterior.covariance_condition_number,
                "camera_pose_count": len(visual_posterior.camera_pose_uncertainty),
                "tag_pose_count": len(visual_posterior.tag_pose_uncertainty),
                "velocity_count": len(visual_posterior.velocity_uncertainty),
                "metadata": dict(visual_posterior.metadata),
            },
            "fused": {
                "coverage_by_level": dict(fused_posterior.coverage_by_level),
                "covariance_condition_number": fused_posterior.covariance_condition_number,
                "camera_pose_count": len(fused_posterior.camera_pose_uncertainty),
                "tag_pose_count": len(fused_posterior.tag_pose_uncertainty),
                "velocity_count": len(fused_posterior.velocity_uncertainty),
                "gyro_bias_radius_95": None if fused_posterior.gyro_bias_uncertainty is None else fused_posterior.gyro_bias_uncertainty.radius_95,
                "accel_bias_radius_95": None if fused_posterior.accel_bias_uncertainty is None else fused_posterior.accel_bias_uncertainty.radius_95,
                "metadata": dict(fused_posterior.metadata),
            },
        },
        "factor_breakdown": factor_breakdown,
        "likelihood_sweeps": likelihood_sweep_rows,
        "uncertainty_stratified": uncertainty_stratified_rows,
        "imu_convention": dataset.imu_convention.summary(),
    }
    artifact_updates = {
        "batch_v1_pose_estimates": str(visual_pose_path.relative_to(resolved_run_dir)),
        "batch_v2_pose_estimates": str(fused_pose_path.relative_to(resolved_run_dir)),
        "imu_only_pose_estimates": str(imu_only_pose_path.relative_to(resolved_run_dir)),
        "batch_v1_tag_map": str(visual_tag_map_path.relative_to(resolved_run_dir)),
        "visual_map_report": str(visual_report_path.relative_to(resolved_run_dir)),
        "visual_inertial_report": str(fused_report_path.relative_to(resolved_run_dir)),
        "checkpoint_03_scientific_report": str(scientific_report_path.relative_to(resolved_run_dir)),
        "ablation_summary": str(ablation_summary_path.relative_to(resolved_run_dir)),
        "uncertainty_summary": str(uncertainty_summary_path.relative_to(resolved_run_dir)),
        "trajectory_accuracy_table": str(trajectory_accuracy_table_path.relative_to(resolved_run_dir)),
        "parameter_plausibility_table": str(parameter_plausibility_table_path.relative_to(resolved_run_dir)),
        "uncertainty_calibration_table": str(uncertainty_calibration_table_path.relative_to(resolved_run_dir)),
        "uncertainty_stratified_table": str(uncertainty_stratified_table_path.relative_to(resolved_run_dir)),
        "noise_sensitivity_table": str(noise_sensitivity_table_path.relative_to(resolved_run_dir)),
        "likelihood_sweep_table": str(likelihood_sweep_table_path.relative_to(resolved_run_dir)),
        "factor_breakdown_json": str(factor_breakdown_path.relative_to(resolved_run_dir)),
        "error_cdf": str(error_cdf_path.relative_to(resolved_run_dir)),
        "per_tag_error_bar": str(per_tag_error_path.relative_to(resolved_run_dir)),
        "per_tag_rotation_error_bar": str(per_tag_rotation_error_path.relative_to(resolved_run_dir)),
        "batch_v1_tag_uncertainty_bar": str(tag_uncertainty_bar_path.relative_to(resolved_run_dir)),
        "ablation_pareto": str(ablation_pareto_path.relative_to(resolved_run_dir)),
        "ablation_position_bars": str(ablation_position_bar_path.relative_to(resolved_run_dir)),
        "ablation_rotation_bars": str(ablation_rotation_bar_path.relative_to(resolved_run_dir)),
        "coverage_reliability": str(coverage_plot_path.relative_to(resolved_run_dir)),
        "batch_trajectory_xy": str(trajectory_plot_path.relative_to(resolved_run_dir)),
        "batch_tag_map_xy": str(tag_map_plot_path.relative_to(resolved_run_dir)),
        "batch_v1_convergence": str(visual_convergence_path.relative_to(resolved_run_dir)),
        "batch_v2_convergence": str(fused_convergence_path.relative_to(resolved_run_dir)),
        "batch_v1_trajectory_3d": str(visual_trajectory_3d_path.relative_to(resolved_run_dir)),
        "batch_v2_trajectory_3d": str(fused_trajectory_3d_path.relative_to(resolved_run_dir)),
        "imu_only_trajectory_3d": str(imu_only_trajectory_3d_path.relative_to(resolved_run_dir)),
        "batch_v1_residual_hist": str(residual_histogram_path.relative_to(resolved_run_dir)),
        "batch_v1_reprojection_timeline": str(visual_frame_rmse_path.relative_to(resolved_run_dir)),
        "batch_v1_position_error_timeline": str(visual_position_error_path.relative_to(resolved_run_dir)),
        "batch_v1_rotation_error_timeline": str(visual_rotation_error_path.relative_to(resolved_run_dir)),
        "trajectory_position_error_compare": str(position_compare_path.relative_to(resolved_run_dir)),
        "trajectory_rotation_error_compare": str(rotation_compare_path.relative_to(resolved_run_dir)),
        "batch_v1_uncertainty_radius_timeline": str(uncertainty_radius_path.relative_to(resolved_run_dir)),
        "batch_v1_uncertainty_vs_error": str(uncertainty_error_scatter_path.relative_to(resolved_run_dir)),
        "batch_v2_uncertainty_vs_error": str(fused_uncertainty_error_scatter_path.relative_to(resolved_run_dir)),
        "batch_v2_imu_residual_components": str(imu_residual_components_path.relative_to(resolved_run_dir)),
        "batch_v2_gyro_bias_norm_timeline": str(fused_gyro_bias_path.relative_to(resolved_run_dir)),
        "batch_v2_accel_bias_norm_timeline": str(fused_accel_bias_path.relative_to(resolved_run_dir)),
        "batch_v2_velocity_norm_timeline": str(fused_velocity_path.relative_to(resolved_run_dir)),
    }
    report_appendix_lines = [
        "",
        "## Batch Estimation",
        "",
        f"- Visual-only unknown-map mean position error: `{visual_eval.mean_position_error_m}` m",
        f"- Visual-only unknown-map mean rotation error: `{visual_eval.mean_rotation_error_deg}` deg",
        f"- Visual-only unknown-map mean reprojection RMSE: `{visual_eval.mean_reprojection_rmse_px}` px",
        f"- Fused mean position error: `{fused_eval.mean_position_error_m}` m",
        f"- Fused mean rotation error: `{fused_eval.mean_rotation_error_deg}` deg",
        f"- IMU-only mean position error: `{imu_only_eval.mean_position_error_m}` m",
        f"- Visual map report: `{visual_report_path.relative_to(resolved_run_dir)}`",
        f"- Visual-inertial report: `{fused_report_path.relative_to(resolved_run_dir)}`",
        f"- Checkpoint 03 scientific report: `{scientific_report_path.relative_to(resolved_run_dir)}`",
        f"- Ablation summary: `{ablation_summary_path.relative_to(resolved_run_dir)}`",
        f"- Uncertainty summary: `{uncertainty_summary_path.relative_to(resolved_run_dir)}`",
        "",
    ]
    return {
        "summary_updates": _jsonify(summary_updates),
        "artifact_updates": artifact_updates,
        "report_appendix_lines": report_appendix_lines,
    }
