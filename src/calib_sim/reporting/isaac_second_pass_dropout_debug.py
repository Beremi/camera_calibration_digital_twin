"""Second-pass fused-dropout debug summaries and bundle builder."""

from __future__ import annotations

import csv
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from calib_sim.reporting.isaac_report_metrics import compute_isaac_estimator_quality, compute_isaac_run_metrics
from calib_sim.reporting.isaac_suppression_windows import generate_suppression_window_artifacts


DEFAULT_SECOND_PASS_DROPOUT_DEBUG_DOC = Path("docs/isaac_second_pass_dropout_debug.md")
DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS = (
    {
        "run_id": "second_pass_dropout_debug_visual_seed_007",
        "debug_mode": "visual_reference",
        "estimator_mode": "visual",
    },
    {
        "run_id": "second_pass_dropout_debug_fused_seed_007",
        "debug_mode": "fused_baseline",
        "estimator_mode": "fused",
    },
    {
        "run_id": "second_pass_dropout_debug_fused_no_reacq_seed_007",
        "debug_mode": "fused_no_reacquisition",
        "estimator_mode": "fused",
    },
    {
        "run_id": "second_pass_dropout_debug_fused_no_imu_during_suppression_seed_007",
        "debug_mode": "fused_no_imu_during_suppression",
        "estimator_mode": "fused",
    },
    {
        "run_id": "second_pass_dropout_debug_fused_covinfl_seed_007",
        "debug_mode": "fused_reacquisition_covariance_inflation",
        "estimator_mode": "fused",
    },
    {
        "run_id": "second_pass_dropout_debug_fused_clipcorr_seed_007",
        "debug_mode": "fused_reacquisition_correction_clipping",
        "estimator_mode": "fused",
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def _bool_from_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _summary_value(metrics: dict[str, Any], *keys: str) -> float | None:
    payload: Any = metrics
    for key in keys:
        if not isinstance(payload, dict):
            return None
        payload = payload.get(key)
    return _float_or_none(payload)


def _load_run_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "analysis" / "metrics.json"
    if metrics_path.exists():
        return _load_json(metrics_path)
    metrics = compute_isaac_run_metrics(run_dir)
    _write_json(metrics_path, metrics)
    return metrics


def _load_run_quality(run_dir: Path) -> dict[str, Any]:
    quality_path = run_dir / "analysis" / "estimator_quality.json"
    if quality_path.exists():
        return _load_json(quality_path)
    quality = compute_isaac_estimator_quality(run_dir)
    _write_json(quality_path, quality)
    return quality


def _visibility_intervals(run_dir: Path, frames: list[dict[str, str]]) -> list[tuple[float, float]]:
    visibility_path = run_dir / "config_snapshot" / "visibility.json"
    if visibility_path.exists():
        payload = _load_json(visibility_path)
        raw_intervals = payload.get("suppressed_intervals_s", [])
        intervals: list[tuple[float, float]] = []
        for interval in raw_intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) < 2:
                continue
            start_s = float(interval[0])
            end_s = float(interval[1])
            if end_s > start_s:
                intervals.append((start_s, end_s))
        if intervals:
            return intervals
    active_times = [
        _float_or_none(row.get("timestamp_s"))
        for row in frames
        if _bool_from_value(row.get("suppression_active"))
    ]
    usable = [float(value) for value in active_times if value is not None]
    if not usable:
        return []
    intervals: list[tuple[float, float]] = []
    start = usable[0]
    previous = usable[0]
    for current in usable[1:]:
        if current - previous > 0.2:
            intervals.append((float(start), float(previous)))
            start = current
        previous = current
    intervals.append((float(start), float(previous)))
    return intervals


def _max_float(values: list[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return float(max(usable))


def _row_nearest_before(rows: list[dict[str, str]], timestamp_s: float) -> dict[str, str] | None:
    candidate = None
    candidate_timestamp = None
    for row in rows:
        row_timestamp = _float_or_none(row.get("timestamp_s"))
        if row_timestamp is None or row_timestamp >= float(timestamp_s):
            continue
        if candidate is None or row_timestamp > float(candidate_timestamp):
            candidate = row
            candidate_timestamp = row_timestamp
    return candidate


def _row_nearest_after(rows: list[dict[str, str]], timestamp_s: float) -> dict[str, str] | None:
    candidate = None
    candidate_timestamp = None
    for row in rows:
        row_timestamp = _float_or_none(row.get("timestamp_s"))
        if row_timestamp is None or row_timestamp < float(timestamp_s):
            continue
        if candidate is None or row_timestamp < float(candidate_timestamp):
            candidate = row
            candidate_timestamp = row_timestamp
    return candidate


def _position_error_jump_after_reacquisition(
    frames: list[dict[str, str]],
    events: list[dict[str, str]],
) -> tuple[float | None, float | None, float | None]:
    accepted_reacq = [
        row
        for row in events
        if _bool_from_value(row.get("accepted")) and _bool_from_value(row.get("is_reacquisition"))
    ]
    if not accepted_reacq:
        return None, None, None
    event = accepted_reacq[0]
    timestamp_s = _float_or_none(event.get("timestamp_s"))
    if timestamp_s is None:
        return None, None, None
    before = _row_nearest_before(frames, float(timestamp_s))
    after = _row_nearest_after(frames, float(timestamp_s))
    before_error = None if before is None else _float_or_none(before.get("position_error_norm_m"))
    after_error = None if after is None else _float_or_none(after.get("position_error_norm_m"))
    if before_error is None or after_error is None:
        return before_error, after_error, None
    return before_error, after_error, float(after_error - before_error)


def _monotonic_growth_during_suppression(frames: list[dict[str, str]], intervals: list[tuple[float, float]]) -> bool:
    monotonic_windows: list[bool] = []
    for start_s, end_s in intervals:
        errors = [
            _float_or_none(row.get("position_error_norm_m"))
            for row in frames
            if _float_or_none(row.get("timestamp_s")) is not None
            and float(start_s) <= float(_float_or_none(row.get("timestamp_s"))) <= float(end_s)
        ]
        usable = [float(value) for value in errors if value is not None]
        if len(usable) < 2:
            continue
        monotonic_windows.append(all(current + 1e-6 >= previous for previous, current in pairwise(usable)))
    return bool(monotonic_windows) and all(monotonic_windows)


def _root_cause_hint(
    *,
    mean_position_error_m: float | None,
    max_position_error_during_suppression_m: float | None,
    coverage_percent: float | None,
    pose_nees: float | None,
    max_bias_norm: float | None,
    bias_growth_ratio: float | None,
    before_reacq_error_m: float | None,
    after_reacq_error_m: float | None,
    jump_after_reacq_m: float | None,
    monotonic_during_suppression: bool,
) -> str:
    if (
        mean_position_error_m is not None
        and mean_position_error_m < 0.05
        and coverage_percent is not None
        and coverage_percent > 80.0
        and pose_nees is not None
        and pose_nees < 20.0
    ):
        return "stable_reference"
    if (
        before_reacq_error_m is not None
        and before_reacq_error_m < 0.05
        and after_reacq_error_m is not None
        and after_reacq_error_m > 0.20
        and jump_after_reacq_m is not None
        and jump_after_reacq_m > 0.15
    ):
        return "reacquisition_update_problem"
    if (
        bias_growth_ratio is not None
        and bias_growth_ratio > 10.0
        and max_bias_norm is not None
        and max_bias_norm > 0.0
        and max_position_error_during_suppression_m is not None
        and max_position_error_during_suppression_m > 0.20
    ):
        return "bias_handling_problem"
    if (
        max_position_error_during_suppression_m is not None
        and max_position_error_during_suppression_m > 0.20
        and monotonic_during_suppression
        and (jump_after_reacq_m is None or jump_after_reacq_m < 0.10)
    ):
        return "propagation_process_problem"
    return "undetermined"


def classify_second_pass_dropout_root_cause(summary: dict[str, Any]) -> str:
    return _root_cause_hint(
        mean_position_error_m=_float_or_none(summary.get("mean_position_error_m")),
        max_position_error_during_suppression_m=_float_or_none(summary.get("max_position_error_during_suppression_m")),
        coverage_percent=_float_or_none(summary.get("empirical_95_coverage_percent")),
        pose_nees=_float_or_none(summary.get("pose_nees")),
        max_bias_norm=_float_or_none(summary.get("max_bias_norm")),
        bias_growth_ratio=_float_or_none(summary.get("bias_growth_ratio")),
        before_reacq_error_m=_float_or_none(summary.get("before_reacquisition_position_error_m")),
        after_reacq_error_m=_float_or_none(summary.get("after_reacquisition_position_error_m")),
        jump_after_reacq_m=_float_or_none(summary.get("jump_after_reacquisition_m")),
        monotonic_during_suppression=bool(summary.get("monotonic_during_suppression", False)),
    )


def _plot_canvas(title: str, y_label: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas = np.full((420, 1080, 3), 249, dtype=np.uint8)
    left, right, top, bottom = 72, 36, 56, 52
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (22, 26, 32), 2, cv2.LINE_AA)
    cv2.putText(canvas, y_label, (24, 404), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (82, 90, 98), 1, cv2.LINE_AA)
    origin_x = left
    origin_y = 420 - bottom
    cv2.line(canvas, (origin_x, top), (origin_x, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    cv2.line(canvas, (origin_x, origin_y), (1080 - right, origin_y), (160, 166, 172), 1, cv2.LINE_AA)
    return canvas, (left, right, top, bottom)


def _save_text_figure(path: Path, *, title: str, lines: list[str]) -> None:
    canvas = np.full((420, 1080, 3), 252, dtype=np.uint8)
    cv2.putText(canvas, title, (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 26, 32), 2, cv2.LINE_AA)
    y = 92
    for line in lines:
        cv2.putText(canvas, line[:120], (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (62, 66, 72), 1, cv2.LINE_AA)
        y += 34
    cv2.imwrite(str(path), canvas)


def _save_timeline_plot(
    path: Path,
    *,
    title: str,
    y_label: str,
    series: list[tuple[str, tuple[int, int, int], list[float], list[float]]],
    intervals: list[tuple[float, float]],
) -> None:
    valid_series = [(label, color, xs, ys) for label, color, xs, ys in series if xs and ys and len(xs) == len(ys)]
    all_x = [float(x) for _, _, xs, _ in valid_series for x in xs]
    all_y = [float(y) for _, _, _, ys in valid_series for y in ys if np.isfinite(float(y))]
    if not all_x or not all_y:
        _save_text_figure(path, title=title, lines=["No finite timeline data available."])
        return
    canvas, (left, right, top, bottom) = _plot_canvas(title, y_label)
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
        x_px = origin_x + int(round((float(x_value) - min_x) / (max_x - min_x) * plot_width))
        y_px = origin_y - int(round((float(y_value) - min_y) / (max_y - min_y) * plot_height))
        return x_px, y_px

    for start_s, end_s in intervals:
        x0, _ = project(start_s, min_y)
        x1, _ = project(end_s, min_y)
        cv2.rectangle(canvas, (min(x0, x1), top), (max(x0, x1), origin_y), (230, 236, 245), -1)
    for label, color, xs, ys in valid_series:
        points = [project(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(float(y))]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        elif len(points) == 1:
            cv2.circle(canvas, points[0], 3, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (780, 36 + 22 * (valid_series.index((label, color, xs, ys)) + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), canvas)


def _summary_paths(run_dir: Path) -> dict[str, Path]:
    analysis_dir = run_dir / "analysis"
    return {
        "summary": analysis_dir / "dropout_debug_summary.json",
        "pose_error": analysis_dir / "dropout_pose_error_timeline.png",
        "velocity": analysis_dir / "dropout_velocity_norm_timeline.png",
        "bias": analysis_dir / "dropout_bias_norm_timeline.png",
        "covariance_trace": analysis_dir / "dropout_covariance_trace_timeline.png",
        "covariance_eigs": analysis_dir / "dropout_covariance_eigs_timeline.png",
        "anchor_innovation": analysis_dir / "dropout_anchor_innovation_timeline.png",
        "relocalization_correction": analysis_dir / "dropout_relocalization_correction_timeline.png",
        "visibility_overlay": analysis_dir / "dropout_visibility_schedule_overlay.png",
    }


def _run_debug_mode(run_id: str) -> str:
    for spec in DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS:
        if spec["run_id"] == run_id:
            return str(spec["debug_mode"])
    return "custom"


def generate_second_pass_dropout_debug_artifacts(run_dir: str | Path) -> dict[str, Any]:
    resolved = Path(run_dir).resolve()
    analysis_dir = resolved / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frames = _read_csv_rows(resolved / "raw" / "dropout_debug_frames.csv")
    events = _read_csv_rows(resolved / "raw" / "dropout_debug_events.csv")
    metrics = _load_run_metrics(resolved)
    quality = _load_run_quality(resolved)
    intervals = _visibility_intervals(resolved, frames)

    suppression_errors = [
        _float_or_none(row.get("position_error_norm_m"))
        for row in frames
        if _bool_from_value(row.get("suppression_active"))
    ]
    max_position_error_during_suppression_m = _max_float(suppression_errors)
    reacquisition_corrections = [
        _float_or_none(row.get("relocalization_correction_norm_m"))
        for row in events
        if _bool_from_value(row.get("accepted")) and _bool_from_value(row.get("is_reacquisition"))
    ]
    max_correction_norm_at_reacquisition_m = _max_float(reacquisition_corrections)
    covariance_traces = [_float_or_none(row.get("covariance_trace")) for row in frames]
    max_covariance_trace = _max_float(covariance_traces)
    gyro_bias_norms = [_float_or_none(row.get("gyro_bias_norm_rps")) for row in frames]
    accel_bias_norms = [_float_or_none(row.get("accel_bias_norm_mps2")) for row in frames]
    max_bias_norm = _max_float(
        [None if value is None else float(value) for value in gyro_bias_norms]
        + [None if value is None else float(value) for value in accel_bias_norms]
    )
    initial_bias_candidates = [
        value
        for value in (
            (gyro_bias_norms[0] if gyro_bias_norms else None),
            (accel_bias_norms[0] if accel_bias_norms else None),
        )
        if value not in (None, "")
    ]
    initial_bias_norm = None if not initial_bias_candidates else float(max(float(value) for value in initial_bias_candidates))
    bias_growth_ratio = None
    if initial_bias_norm not in (None, 0.0) and max_bias_norm is not None:
        bias_growth_ratio = float(max_bias_norm / max(initial_bias_norm, 1e-9))
    before_reacq_error_m, after_reacq_error_m, jump_after_reacq_m = _position_error_jump_after_reacquisition(
        frames,
        events,
    )
    monotonic_during_suppression = _monotonic_growth_during_suppression(frames, intervals)

    mean_position_error_m = _summary_value(metrics, "trajectory", "mean_position_error_m")
    mean_waypoint_error_m = _summary_value(metrics, "control", "mean_waypoint_error_m")
    completion_fraction = _summary_value(metrics, "control", "completion_fraction")
    empirical_95_coverage_percent = _summary_value(metrics, "uncertainty_calibration", "empirical_95_coverage_percent")
    pose_nees = _summary_value(metrics, "uncertainty_calibration", "pose_nees")
    root_cause_hint = _root_cause_hint(
        mean_position_error_m=mean_position_error_m,
        max_position_error_during_suppression_m=max_position_error_during_suppression_m,
        coverage_percent=empirical_95_coverage_percent,
        pose_nees=pose_nees,
        max_bias_norm=max_bias_norm,
        bias_growth_ratio=bias_growth_ratio,
        before_reacq_error_m=before_reacq_error_m,
        after_reacq_error_m=after_reacq_error_m,
        jump_after_reacq_m=jump_after_reacq_m,
        monotonic_during_suppression=bool(monotonic_during_suppression),
    )
    blocker_clear = bool(
        mean_position_error_m is not None
        and mean_position_error_m < 0.05
        and empirical_95_coverage_percent is not None
        and empirical_95_coverage_percent > 80.0
        and pose_nees is not None
        and pose_nees < 20.0
        and completion_fraction is not None
        and abs(completion_fraction - 1.0) < 1e-9
        and (jump_after_reacq_m is None or jump_after_reacq_m < 0.20)
    )

    summary = {
        "run_id": resolved.name,
        "run_dir": str(resolved),
        "debug_mode": _run_debug_mode(resolved.name),
        "mean_position_error_m": mean_position_error_m,
        "mean_waypoint_error_m": mean_waypoint_error_m,
        "completion_fraction": completion_fraction,
        "empirical_95_coverage_percent": empirical_95_coverage_percent,
        "pose_nees": pose_nees,
        "max_position_error_during_suppression_m": max_position_error_during_suppression_m,
        "max_correction_norm_at_reacquisition_m": max_correction_norm_at_reacquisition_m,
        "max_covariance_trace": max_covariance_trace,
        "max_bias_norm": max_bias_norm,
        "bias_growth_ratio": bias_growth_ratio,
        "before_reacquisition_position_error_m": before_reacq_error_m,
        "after_reacquisition_position_error_m": after_reacq_error_m,
        "jump_after_reacquisition_m": jump_after_reacq_m,
        "monotonic_during_suppression": bool(monotonic_during_suppression),
        "root_cause_hint": root_cause_hint,
        "blocker_clear": blocker_clear,
        "suppression_intervals_s": [[float(start_s), float(end_s)] for start_s, end_s in intervals],
    }

    paths = _summary_paths(resolved)
    _write_json(paths["summary"], summary)

    frame_times = [float(_float_or_none(row.get("timestamp_s")) or 0.0) for row in frames]
    _save_timeline_plot(
        paths["pose_error"],
        title="Dropout Pose Error Timeline",
        y_label="position error [m]",
        series=[
            (
                "position error",
                (53, 102, 188),
                frame_times,
                [float(_float_or_none(row.get("position_error_norm_m")) or 0.0) for row in frames],
            )
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["velocity"],
        title="Dropout Velocity Norm Timeline",
        y_label="velocity norm [m/s]",
        series=[
            (
                "velocity norm",
                (53, 102, 188),
                frame_times,
                [float(_float_or_none(row.get("velocity_norm_mps")) or 0.0) for row in frames],
            )
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["bias"],
        title="Dropout Bias Norm Timeline",
        y_label="bias norm",
        series=[
            (
                "gyro bias",
                (53, 102, 188),
                frame_times,
                [float(_float_or_none(row.get("gyro_bias_norm_rps")) or 0.0) for row in frames],
            ),
            (
                "accel bias",
                (188, 92, 60),
                frame_times,
                [float(_float_or_none(row.get("accel_bias_norm_mps2")) or 0.0) for row in frames],
            ),
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["covariance_trace"],
        title="Dropout Covariance Trace Timeline",
        y_label="trace",
        series=[
            (
                "covariance trace",
                (53, 102, 188),
                frame_times,
                [float(_float_or_none(row.get("covariance_trace")) or 0.0) for row in frames],
            )
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["covariance_eigs"],
        title="Dropout Covariance Eigenvalue Timeline",
        y_label="eigenvalue",
        series=[
            (
                "min eig",
                (53, 102, 188),
                frame_times,
                [float(_float_or_none(row.get("covariance_min_eigenvalue")) or 0.0) for row in frames],
            ),
            (
                "max eig",
                (188, 92, 60),
                frame_times,
                [float(_float_or_none(row.get("covariance_max_eigenvalue")) or 0.0) for row in frames],
            ),
        ],
        intervals=intervals,
    )
    event_times = [float(_float_or_none(row.get("timestamp_s")) or 0.0) for row in events]
    _save_timeline_plot(
        paths["anchor_innovation"],
        title="Dropout Anchor Innovation Timeline",
        y_label="innovation",
        series=[
            (
                "pose innovation",
                (53, 102, 188),
                event_times,
                [float(_float_or_none(row.get("pose_innovation_norm_m")) or 0.0) for row in events],
            ),
            (
                "orientation innovation",
                (188, 92, 60),
                event_times,
                [float(_float_or_none(row.get("orientation_innovation_norm_deg")) or 0.0) for row in events],
            ),
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["relocalization_correction"],
        title="Dropout Reacquisition Correction Timeline",
        y_label="correction [m]",
        series=[
            (
                "reacquisition correction",
                (53, 102, 188),
                event_times,
                [float(_float_or_none(row.get("relocalization_correction_norm_m")) or 0.0) for row in events],
            )
        ],
        intervals=intervals,
    )
    _save_timeline_plot(
        paths["visibility_overlay"],
        title="Dropout Visibility Schedule Overlay",
        y_label="visibility state",
        series=[
            (
                "raw visible",
                (53, 102, 188),
                frame_times,
                [1.0 if _bool_from_value(row.get("anchor_visible_raw")) else 0.0 for row in frames],
            ),
            (
                "effective visible",
                (188, 92, 60),
                frame_times,
                [1.0 if _bool_from_value(row.get("anchor_visible_effective")) else 0.0 for row in frames],
            ),
            (
                "suppressed",
                (84, 146, 62),
                frame_times,
                [1.0 if _bool_from_value(row.get("suppression_active")) else 0.0 for row in frames],
            ),
        ],
        intervals=intervals,
    )

    payload = {
        "run_id": resolved.name,
        "summary": summary,
        "metrics_path": str((resolved / "analysis" / "metrics.json").resolve()),
        "quality_path": str((resolved / "analysis" / "estimator_quality.json").resolve()),
        "frames_path": str((resolved / "raw" / "dropout_debug_frames.csv").resolve()),
        "events_path": str((resolved / "raw" / "dropout_debug_events.csv").resolve()),
        "figure_paths": {key: str(path.resolve()) for key, path in paths.items() if key != "summary"},
        "suppression_window_artifacts": generate_suppression_window_artifacts(resolved),
    }
    _write_json(paths["summary"], payload)
    return payload


def build_second_pass_dropout_debug_bundle(
    output_root: str | Path,
    *,
    docs_path: str | Path = DEFAULT_SECOND_PASS_DROPOUT_DEBUG_DOC,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    bundle_dir = root / "latest_second_pass_dropout_debug"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    for spec in DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS:
        run_dir = root / str(spec["run_id"])
        summary_path = run_dir / "analysis" / "dropout_debug_summary.json"
        if not summary_path.exists():
            continue
        payload = _load_json(summary_path)
        summary = dict(payload.get("summary", {}))
        row = {
            "run_id": str(spec["run_id"]),
            "debug_mode": str(spec["debug_mode"]),
            "mean_position_error_m": summary.get("mean_position_error_m"),
            "mean_waypoint_error_m": summary.get("mean_waypoint_error_m"),
            "completion_fraction": summary.get("completion_fraction"),
            "empirical_95_coverage_percent": summary.get("empirical_95_coverage_percent"),
            "pose_nees": summary.get("pose_nees"),
            "max_position_error_during_suppression_m": summary.get("max_position_error_during_suppression_m"),
            "max_correction_norm_at_reacquisition_m": summary.get("max_correction_norm_at_reacquisition_m"),
            "max_covariance_trace": summary.get("max_covariance_trace"),
            "max_bias_norm": summary.get("max_bias_norm"),
            "root_cause_hint": summary.get("root_cause_hint"),
            "blocker_clear": summary.get("blocker_clear"),
        }
        rows.append(row)
        detailed_rows.append({"run_id": str(spec["run_id"]), **summary})
    summary_csv = bundle_dir / "summary.csv"
    summary_json = bundle_dir / "summary.json"
    _write_csv(summary_csv, rows)
    bundle_payload = {
        "bundle_dir": str(bundle_dir),
        "rows": rows,
        "detailed_rows": detailed_rows,
        "blocker_clear_run_ids": [row["run_id"] for row in rows if bool(row.get("blocker_clear"))],
    }
    _write_json(summary_json, bundle_payload)

    docs_target = Path(docs_path).resolve()
    docs_target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Isaac Second-Pass Dropout Debug Pack",
        "",
        "This bundle tracks the seed-007 fused intermittent-anchor blocker before the second-pass draft suite is rerun.",
        "",
        f"- summary csv: `{summary_csv}`",
        f"- summary json: `{summary_json}`",
        "",
        "## Runs",
        "",
    ]
    if not rows:
        lines.extend(
            [
                "No debug summaries have been generated yet.",
                "",
                "Run `scripts/run_isaac_second_pass_dropout_debug.py --headless --execute-missing` to populate this pack.",
            ]
        )
    else:
        for row in rows:
            lines.append(
                "- "
                + ", ".join(
                    [
                        f"`{row['run_id']}`",
                        f"mode={row['debug_mode']}",
                        f"pos={row['mean_position_error_m']}",
                        f"way={row['mean_waypoint_error_m']}",
                        f"coverage={row['empirical_95_coverage_percent']}",
                        f"nees={row['pose_nees']}",
                        f"hint={row['root_cause_hint']}",
                    ]
                )
            )
        lines.extend(
            [
                "",
                "## Current Read",
                "",
                f"- blocker-clear runs: {', '.join(bundle_payload['blocker_clear_run_ids']) if bundle_payload['blocker_clear_run_ids'] else 'none yet'}",
                "- do not rerun the 18-run second-pass suite or rebuild the second-pass draft PDF until at least one fused intermittent-anchor variant clears the interim blocker bar.",
            ]
        )
    docs_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "bundle_dir": str(bundle_dir.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "summary_json": str(summary_json.resolve()),
        "docs_path": str(docs_target),
        "rows": rows,
    }


__all__ = [
    "DEFAULT_SECOND_PASS_DROPOUT_DEBUG_DOC",
    "DEFAULT_SECOND_PASS_DROPOUT_DEBUG_RUN_SPECS",
    "build_second_pass_dropout_debug_bundle",
    "classify_second_pass_dropout_root_cause",
    "generate_second_pass_dropout_debug_artifacts",
]
