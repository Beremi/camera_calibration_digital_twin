"""Suppression-window summaries for fused intermittent-anchor debug runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
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
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _rows_in_window(rows: list[dict[str, str]], *, start_s: float, end_s: float) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if _float_or_none(row.get("timestamp_s")) is not None
        and float(start_s) <= float(_float_or_none(row.get("timestamp_s"))) <= float(end_s)
    ]


def _row_before(rows: list[dict[str, str]], *, timestamp_s: float) -> dict[str, str] | None:
    candidate: dict[str, str] | None = None
    candidate_timestamp: float | None = None
    for row in rows:
        row_timestamp = _float_or_none(row.get("timestamp_s"))
        if row_timestamp is None or row_timestamp > float(timestamp_s):
            continue
        if candidate is None or row_timestamp > float(candidate_timestamp):
            candidate = row
            candidate_timestamp = row_timestamp
    return candidate


def _row_after(rows: list[dict[str, str]], *, timestamp_s: float) -> dict[str, str] | None:
    candidate: dict[str, str] | None = None
    candidate_timestamp: float | None = None
    for row in rows:
        row_timestamp = _float_or_none(row.get("timestamp_s"))
        if row_timestamp is None or row_timestamp < float(timestamp_s):
            continue
        if candidate is None or row_timestamp < float(candidate_timestamp):
            candidate = row
            candidate_timestamp = row_timestamp
    return candidate


def _first_accepted_reacquisition(
    events: list[dict[str, str]],
    *,
    end_s: float,
) -> dict[str, str] | None:
    candidates = [
        row
        for row in events
        if _bool_from_value(row.get("accepted"))
        and _bool_from_value(row.get("is_reacquisition"))
        and _float_or_none(row.get("timestamp_s")) is not None
        and float(_float_or_none(row.get("timestamp_s"))) >= float(end_s)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: float(_float_or_none(row.get("timestamp_s"))))


def _safe_growth(end_value: float | None, start_value: float | None) -> float | None:
    if end_value is None or start_value is None:
        return None
    return float(end_value - start_value)


def _safe_slope(end_value: float | None, start_value: float | None, duration_s: float) -> float | None:
    growth = _safe_growth(end_value, start_value)
    if growth is None or duration_s <= 1e-9:
        return None
    return float(growth / duration_s)


def _growth_ratio(end_value: float | None, start_value: float | None) -> float | None:
    if end_value is None or start_value is None:
        return None
    if abs(float(start_value)) <= 1e-9:
        return None if abs(float(end_value)) <= 1e-9 else float("inf")
    return float(end_value / start_value)


def classify_suppression_window(row: dict[str, Any]) -> str:
    error_growth = _float_or_none(row.get("position_error_growth_m"))
    bias_growth_ratio = max(
        [
            value
            for value in (
                _float_or_none(row.get("gyro_bias_growth_ratio")),
                _float_or_none(row.get("accel_bias_growth_ratio")),
            )
            if value is not None
        ],
        default=None,
    )
    reacq_jump_m = _float_or_none(row.get("post_reacquisition_position_error_jump_m"))
    covariance_growth = _float_or_none(row.get("covariance_trace_growth"))
    if bias_growth_ratio is not None and bias_growth_ratio > 5.0 and error_growth is not None and error_growth > 0.02:
        return "bias_growth_dominant"
    if error_growth is not None and error_growth > 0.02:
        return "mean_state_drift_dominant"
    if reacq_jump_m is not None and reacq_jump_m > max(0.05, abs(float(error_growth or 0.0)) * 0.5):
        return "reacquisition_jump_dominant"
    if covariance_growth is not None and covariance_growth > 0.0:
        return "covariance_only"
    return "covariance_only"


def generate_suppression_window_artifacts(run_dir: str | Path) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir).resolve()
    analysis_dir = resolved_run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frames = _read_csv_rows(resolved_run_dir / "raw" / "dropout_debug_frames.csv")
    events = _read_csv_rows(resolved_run_dir / "raw" / "dropout_debug_events.csv")
    intervals = _visibility_intervals(resolved_run_dir, frames)
    window_rows: list[dict[str, Any]] = []
    classifier_counts = {
        "mean_state_drift_dominant": 0,
        "bias_growth_dominant": 0,
        "reacquisition_jump_dominant": 0,
        "covariance_only": 0,
    }
    for index, (start_s, end_s) in enumerate(intervals):
        window_frames = _rows_in_window(frames, start_s=float(start_s), end_s=float(end_s))
        if not window_frames:
            continue
        start_row = window_frames[0]
        end_row = window_frames[-1]
        duration_s = max(float(end_s) - float(start_s), 1e-9)
        start_position_error_m = _float_or_none(start_row.get("position_error_norm_m"))
        end_position_error_m = _float_or_none(end_row.get("position_error_norm_m"))
        start_velocity_norm_mps = _float_or_none(start_row.get("velocity_norm_mps"))
        end_velocity_norm_mps = _float_or_none(end_row.get("velocity_norm_mps"))
        start_covariance_trace = _float_or_none(start_row.get("covariance_trace"))
        end_covariance_trace = _float_or_none(end_row.get("covariance_trace"))
        start_covariance_min_eigenvalue = _float_or_none(start_row.get("covariance_min_eigenvalue"))
        end_covariance_min_eigenvalue = _float_or_none(end_row.get("covariance_min_eigenvalue"))
        start_covariance_max_eigenvalue = _float_or_none(start_row.get("covariance_max_eigenvalue"))
        end_covariance_max_eigenvalue = _float_or_none(end_row.get("covariance_max_eigenvalue"))
        start_gyro_bias_norm_rps = _float_or_none(start_row.get("gyro_bias_norm_rps"))
        end_gyro_bias_norm_rps = _float_or_none(end_row.get("gyro_bias_norm_rps"))
        start_accel_bias_norm_mps2 = _float_or_none(start_row.get("accel_bias_norm_mps2"))
        end_accel_bias_norm_mps2 = _float_or_none(end_row.get("accel_bias_norm_mps2"))
        imu_packets_used_for_prediction = sum(
            int(_float_or_none(row.get("imu_packets_used_for_prediction")) or 0.0) for row in window_frames
        )
        imu_packets_rejected_for_prediction = sum(
            int(_float_or_none(row.get("imu_packets_rejected_for_prediction")) or 0.0) for row in window_frames
        )
        reacq_event = _first_accepted_reacquisition(events, end_s=float(end_s))
        reacq_timestamp_s = None if reacq_event is None else _float_or_none(reacq_event.get("timestamp_s"))
        post_reacq_row = None if reacq_timestamp_s is None else _row_after(frames, timestamp_s=float(reacq_timestamp_s))
        post_reacq_position_error_m = (
            None if post_reacq_row is None else _float_or_none(post_reacq_row.get("position_error_norm_m"))
        )
        row = {
            "window_index": int(index),
            "start_s": float(start_s),
            "end_s": float(end_s),
            "start_position_error_m": start_position_error_m,
            "end_position_error_m": end_position_error_m,
            "position_error_growth_m": _safe_growth(end_position_error_m, start_position_error_m),
            "position_error_slope_mps": _safe_slope(end_position_error_m, start_position_error_m, duration_s),
            "start_velocity_norm_mps": start_velocity_norm_mps,
            "end_velocity_norm_mps": end_velocity_norm_mps,
            "velocity_norm_growth_mps": _safe_growth(end_velocity_norm_mps, start_velocity_norm_mps),
            "start_covariance_trace": start_covariance_trace,
            "end_covariance_trace": end_covariance_trace,
            "covariance_trace_growth": _safe_growth(end_covariance_trace, start_covariance_trace),
            "start_covariance_min_eigenvalue": start_covariance_min_eigenvalue,
            "end_covariance_min_eigenvalue": end_covariance_min_eigenvalue,
            "covariance_min_eigenvalue_growth": _safe_growth(
                end_covariance_min_eigenvalue,
                start_covariance_min_eigenvalue,
            ),
            "start_covariance_max_eigenvalue": start_covariance_max_eigenvalue,
            "end_covariance_max_eigenvalue": end_covariance_max_eigenvalue,
            "covariance_max_eigenvalue_growth": _safe_growth(
                end_covariance_max_eigenvalue,
                start_covariance_max_eigenvalue,
            ),
            "start_gyro_bias_norm_rps": start_gyro_bias_norm_rps,
            "end_gyro_bias_norm_rps": end_gyro_bias_norm_rps,
            "gyro_bias_norm_growth_rps": _safe_growth(end_gyro_bias_norm_rps, start_gyro_bias_norm_rps),
            "gyro_bias_growth_ratio": _growth_ratio(end_gyro_bias_norm_rps, start_gyro_bias_norm_rps),
            "start_accel_bias_norm_mps2": start_accel_bias_norm_mps2,
            "end_accel_bias_norm_mps2": end_accel_bias_norm_mps2,
            "accel_bias_norm_growth_mps2": _safe_growth(end_accel_bias_norm_mps2, start_accel_bias_norm_mps2),
            "accel_bias_growth_ratio": _growth_ratio(end_accel_bias_norm_mps2, start_accel_bias_norm_mps2),
            "imu_packets_used_for_prediction": int(imu_packets_used_for_prediction),
            "imu_packets_rejected_for_prediction": int(imu_packets_rejected_for_prediction),
            "imu_packets_total_for_prediction": int(
                imu_packets_used_for_prediction + imu_packets_rejected_for_prediction
            ),
            "first_reacquisition_timestamp_s": reacq_timestamp_s,
            "first_reacquisition_innovation_norm_m": None
            if reacq_event is None
            else _float_or_none(reacq_event.get("pose_innovation_norm_m")),
            "first_reacquisition_nis": None if reacq_event is None else _float_or_none(reacq_event.get("anchor_nis")),
            "first_reacquisition_correction_norm_m": None
            if reacq_event is None
            else _float_or_none(reacq_event.get("relocalization_correction_norm_m")),
            "post_reacquisition_position_error_m": post_reacq_position_error_m,
            "post_reacquisition_position_error_jump_m": _safe_growth(
                post_reacq_position_error_m,
                end_position_error_m,
            ),
        }
        window_classifier = classify_suppression_window(row)
        classifier_counts[window_classifier] += 1
        row["window_classifier"] = window_classifier
        window_rows.append(row)
    dominant_classifier = "covariance_only"
    if window_rows:
        dominant_classifier = max(classifier_counts, key=classifier_counts.get)
    summary_payload = {
        "run_dir": str(resolved_run_dir),
        "window_count": len(window_rows),
        "windows": window_rows,
        "classifier_counts": classifier_counts,
        "dominant_classifier": dominant_classifier,
    }
    csv_path = analysis_dir / "suppression_window_summary.csv"
    json_path = analysis_dir / "suppression_window_summary.json"
    note_path = analysis_dir / "suppression_window_note.md"
    _write_csv(csv_path, window_rows)
    _write_json(json_path, summary_payload)
    note_lines = [
        f"# Suppression Window Summary for `{resolved_run_dir.name}`",
        "",
        f"- dominant classifier: `{dominant_classifier}`",
        f"- window count: `{len(window_rows)}`",
        "",
        "## Windows",
        "",
    ]
    for row in window_rows:
        start_error = 0.0 if row["start_position_error_m"] is None else float(row["start_position_error_m"])
        end_error = 0.0 if row["end_position_error_m"] is None else float(row["end_position_error_m"])
        note_lines.append(
            "- "
            f"window {row['window_index']} [{row['start_s']:.2f}, {row['end_s']:.2f}] s: "
            f"classifier=`{row['window_classifier']}`, "
            f"error {start_error:.4f}->{end_error:.4f} m, "
            f"velocity growth={float(row['velocity_norm_growth_mps'] or 0.0):.4f} m/s, "
            f"cov trace growth={float(row['covariance_trace_growth'] or 0.0):.6f}, "
            f"used/rejected IMU={row['imu_packets_used_for_prediction']}/{row['imu_packets_rejected_for_prediction']}, "
            f"first reacq innovation={row['first_reacquisition_innovation_norm_m']}, "
            f"first reacq correction={row['first_reacquisition_correction_norm_m']}"
        )
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    return {
        "run_dir": str(resolved_run_dir),
        "summary_csv": str(csv_path),
        "summary_json": str(json_path),
        "note_path": str(note_path),
        "window_count": len(window_rows),
        "dominant_classifier": dominant_classifier,
    }
